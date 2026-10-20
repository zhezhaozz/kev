"""Microsoft WebQSP labels and NSM graphs. Gold answers never enter an Observation."""
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .api import option_text
from .suite import digest, read_json


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    start: str | None
    topic_name: str


@dataclass(frozen=True)
class Example:
    query: Query
    answer_sets: tuple[tuple[str, ...], ...]

    @property
    def answers(self):
        return {answer for answers in self.answer_sets for answer in answers}


def questions(path):
    """Use Microsoft's quality gate; retain every parse for best-parse scoring."""
    result, excluded, seen = {}, [], set()
    for row in read_json(path)["Questions"]:
        qid = row["QuestionId"]
        if qid in seen:
            raise ValueError(f"duplicate Microsoft question: {qid}")
        seen.add(qid)
        parses = row["Parses"]
        if not any(p["AnnotatorComment"]["QuestionQuality"] == "Good" and
                   p["AnnotatorComment"]["ParseQuality"] == "Complete" for p in parses):
            excluded.append(qid)
            continue
        # Entity linking is an oracle input, like NSM's gold-topic setup. Choose by
        # annotation order, never by whether a graph contains a gold answer.
        topic = next((p for p in parses if p.get("TopicEntityMid")), {})
        query = Query(qid, row["RawQuestion"], topic.get("TopicEntityMid"), topic.get("TopicEntityName") or "")
        answers = tuple(tuple(a["AnswerArgument"] for a in p["Answers"]) for p in parses)
        if not all(isinstance(a, str) for group in answers for a in group):
            raise ValueError(f"{qid}: AnswerArgument must be a string")
        result[qid] = Example(query, answers)
    return result, excluded, seen


class GraphFile:
    """Index only question IDs and byte offsets; load one question graph at a time."""
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = {}
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                qid = row["id"]
                if qid in self.offsets:
                    raise ValueError(f"{path}: duplicate question {qid}")
                self.offsets[qid] = offset

    def get(self, qid):
        if qid not in self.offsets:
            return None
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[qid])
            # Ignore NSM answers, topic annotations and any supervision fields.
            return json.loads(stream.readline())["subgraph"]


@dataclass(frozen=True, order=True)
class Action:
    relation: str
    target: str
    reverse: bool = False


NO_OP = "NO_OP"


class Graph:
    def __init__(self, subgraph, entities, relations, inverse=True):
        def lookup(vocab, index):
            if type(index) is not int or not 0 <= index < len(vocab):
                raise ValueError(f"invalid NSM vocabulary index: {index!r}")
            return vocab[index]

        self.nodes = {lookup(entities, i) for i in subgraph["entities"]}
        edges = defaultdict(set)
        for source, relation, target in subgraph["tuples"]:
            source, target = lookup(entities, source), lookup(entities, target)
            relation = lookup(relations, relation)
            self.nodes.update((source, target))
            edges[source].add(Action(relation, target))
            if inverse:
                edges[target].add(Action(relation, source, reverse=True))
        self.edges = {node: tuple(sorted(actions)) for node, actions in edges.items()}

    def actions(self, current):
        # Keep all legal edges. No gold-dependent filtering, retriever, degree cap
        # or special removal of backtracking edges. NO_OP consumes one hop.
        return (Action(NO_OP, current), *self.edges.get(current, ()))

    def reachable(self, start, hops):
        if start not in self.nodes:
            return set()
        reached, frontier = {start}, {start}
        for _ in range(hops):
            following = {a.target for node in frontier for a in self.edges.get(node, ())} - reached
            reached.update(following)
            frontier = following
        return reached


class Dataset:
    def __init__(self, questions_path, nsm_dir, names_path, split, inverse=True):
        self.split, self.inverse = split, inverse
        root = Path(nsm_dir)
        self.files = {s: GraphFile(root / f"{s}_simple.json") for s in ("train", "dev", "test")}
        ids = {s: set(f.offsets) for s, f in self.files.items()}
        if any(ids[a] & ids[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
            raise ValueError("NSM train/dev/test question IDs overlap")
        raw, excluded, all_ids = questions(questions_path)
        expected_prefix = "WebQTest-" if split == "test" else "WebQTrn-"
        if not all(qid.startswith(expected_prefix) for qid in all_ids):
            raise ValueError(f"{split} requires the original Microsoft {'test' if split == 'test' else 'train'} JSON")
        expected = ids["test"] if split == "test" else ids["train"] | ids["dev"]
        if expected - all_ids:
            raise ValueError(f"NSM has question IDs absent from the Microsoft file: {sorted(expected - all_ids)[:5]}")
        if split != "test" and all_ids - expected:
            raise ValueError("Microsoft training questions are missing from the NSM train/dev split assignment")
        # Test always uses the complete original test set, including missing graphs.
        selected = all_ids if split == "test" else ids[split]
        self.examples = [example for qid, example in raw.items() if qid in selected]
        if not self.examples:
            raise ValueError("no eligible questions in the selected split")
        self.excluded = sorted(set(excluded) & selected)
        self.entities = (root / "entities.txt").read_text(encoding="utf-8").splitlines()
        self.relations = (root / "relations.txt").read_text(encoding="utf-8").splitlines()
        self.names = read_json(names_path)
        if not isinstance(self.names, dict) or not self.names or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in self.names.items()):
            raise ValueError("--names must be an independent JSON object mapping Freebase IDs to names")
        sources = [Path(questions_path), Path(names_path), root / "entities.txt", root / "relations.txt",
                   *(f.path for f in self.files.values())]
        self.provenance = {"split": split, "questions": len(self.examples), "quality_excluded_ids": self.excluded,
                           "files": [{"path": str(p.resolve()), "sha256": digest(p)} for p in sources]}

    def graph(self, example):
        subgraph = self.files[self.split].get(example.query.id)
        return None if subgraph is None else Graph(subgraph, self.entities, self.relations, self.inverse)


def entity_text(entity, names):
    name = names.get(entity)
    return f"{name} [{entity}]" if name and name != entity else entity


def action_text(action, names):
    if action.relation == NO_OP:
        return "Stay at the current entity (NO_OP)"
    direction = "inverse " if action.reverse else ""
    return f"{direction}{action.relation} -> {entity_text(action.target, names)}"


def observation(query, current, path, actions, names, hops):
    """Only question, gold topic, current node, history and legal edge descriptions."""
    history = "\n".join(f"{i + 1}. {action_text(a, names)}" for i, a in enumerate(path)) or "(none)"
    state = (f"Question: {query.text}\nTopic: {query.topic_name} [{query.start}]\n"
             f"Current entity: {entity_text(current, names)}\n"
             f"Moves used: {len(path)} of {hops}\nPath from the topic:\n{history}")
    return {"state": state, "questions": [{
        "instr": ("Choose the next graph move to reach an answer to the question by the final move. "
                  "An inverse relation traverses an edge backward. NO_OP stays at the current entity and uses one move."),
        "options": [option_text(str(i), action_text(a, names)) for i, a in enumerate(actions)],
        "label": 0,  # Encoder schema only; never used as a supervised target.
    }]}


def score(example, predicted):
    """Microsoft's macro best-parse P/R/F1, without its 1e-40 zero-count trick."""
    predicted = set(predicted)
    best = None
    for answers in example.answer_sets:
        if not answers:
            precision, recall, f1 = (0.0, 1.0, 0.0) if predicted else (1.0, 1.0, 1.0)
        elif not predicted:
            precision, recall, f1 = 1.0, 0.0, 0.0
        else:
            correct = sum(answer in predicted for answer in answers)
            incorrect = len(predicted - set(answers))
            precision = correct / (correct + incorrect) if correct + incorrect else 0.0
            recall = correct / len(answers)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = {"precision": precision, "recall": recall, "f1": f1}
        if best is None or f1 > best["f1"]:
            best = candidate
    return best


def graph_diagnostics(example, graph, names, hops):
    answers = example.answers
    reachable = set() if graph is None else graph.reachable(example.query.start, hops)
    mids = set() if graph is None else {node for node in graph.nodes if node.startswith(("m.", "g."))}
    return {"missing_graph": graph is None,
            "missing_start": graph is None or example.query.start not in graph.nodes,
            "reachable_answer": bool(answers & reachable),
            "reachable_answer_recall": len(answers & reachable) / len(answers) if answers else 0.0,
            "entity_ids": len(mids), "named_entity_ids": sum(bool(names.get(mid)) for mid in mids)}
