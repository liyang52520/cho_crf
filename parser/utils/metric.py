# -*- coding: utf-8 -*-

from collections import Counter


class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class SpanF1Metric(Metric):

    def __init__(self, eps=1e-8):
        super(SpanF1Metric, self).__init__()

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            tp = list((Counter(pred) & Counter(gold)).elements())
            self.tp += len(tp)
            self.pred += len(pred)
            self.gold += len(gold)

    def __repr__(self):
        return f"P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)


class LabelMetric(Metric):
    def __init__(self, eps=1e-8):
        super(LabelMetric, self).__init__()

        self.eps = eps
        self.total = 0.0
        self.correct = 0.0

    def __repr__(self):
        return f"Score: {self.score:6.2%}"

    def __call__(self, preds, golds, mask):
        rel_mask = preds.eq(golds)[mask]

        self.total += len(rel_mask)
        self.correct += rel_mask.sum().item()

    @property
    def score(self):
        return self.correct / (self.total + self.eps)

