import unittest
from score_whitening_v2 import score

class ScoringTests(unittest.TestCase):
    def test_abstention_not_detection(self):
        r=score({'1':'whitening'},[{'card_id':'1','decision':'needs_review'}])
        self.assertEqual(r['confusion_counts']['whitening']['whitening_detected'],0)
        self.assertEqual(r['confusion_counts']['whitening']['needs_review'],1)
    def test_missing_labels_counted(self):
        self.assertEqual(score({'1':'clean'},[])['missing_labeled_cards'],['1'])
    def test_consistency_separate_from_accuracy(self):
        r=score({'1':'whitening'},[{'card_id':'1','decision':'clean'}]*3)
        self.assertEqual(r['fully_consistent_cards'],1)
        self.assertEqual(r['confusion_counts']['whitening']['clean'],3)

if __name__=='__main__':unittest.main(verbosity=2)
