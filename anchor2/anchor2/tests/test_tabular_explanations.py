import unittest

from anchor2.anchor2.tabular_explanation import TabularExplanation, \
    EqualityPredicate, InequalityPredicate, GreaterOrEqualPredicate, LessPredicate


class TestTabularExplanations(unittest.TestCase):

    def test_equal(self):
        exp1 = TabularExplanation(None, None, None)
        exp1.predicates = [EqualityPredicate(0.0, 1, "First Feature"), GreaterOrEqualPredicate(-2.0, 2, "Second Feature")]
        exp2 = TabularExplanation(None, None, None)
        exp2.predicates = [GreaterOrEqualPredicate(-2.0, 2, "Second Feature"), EqualityPredicate(0.0, 1, "First Feature")]
        self.assertEqual(exp1, exp2)

    def test_notequal(self):
        exp1, exp2 = TabularExplanation(None, None, None), TabularExplanation(None, None, None)
        exp1.predicates = [EqualityPredicate(0.0, 1, "First Feature"), GreaterOrEqualPredicate(-2.0, 2, "Second Feature")]
        exp2.predicates = [GreaterOrEqualPredicate(-3.0, 2, "Second Feature"), EqualityPredicate(0.0, 1, "First Feature")]
        self.assertNotEqual(exp1, exp2)

    def test_predicate_notequal(self):
        p1 = GreaterOrEqualPredicate(-2.0, 2, "Second Feature")
        p2 = LessPredicate(-2.0, 2, "Second Feature")
        self.assertNotEqual(p1, p2)

    def test_predicate_equal(self):
        p1 = LessPredicate(-2.0, 2, "Third Feature")
        p2 = LessPredicate(-2.0, 2, "Third Feature")
        self.assertEqual(p1, p2)

    def test_predicate_str(self):
        with self.subTest("<<"):
            p1 = LessPredicate(-2.0, 2, "X")
            self.assertEqual(str(p1), "X < -2.0")

        with self.subTest("=="):
            p2 = EqualityPredicate(-2.0, 2, "Y")
            self.assertEqual(str(p2), "Y == -2.0")

        with self.subTest("!="):
            p3 = InequalityPredicate(-2.0, 2, "Z")
            self.assertEqual(str(p3), "Z != -2.0")

        with self.subTest(">="):
            p4 = GreaterOrEqualPredicate(-2.0, 2, "Theta")
            self.assertEqual(str(p4), "Theta >= -2.0")

    def test_explanation_str(self):
        exp = TabularExplanation(None, None, None)
        exp.predicates = [EqualityPredicate(13.0, 1, "X"), GreaterOrEqualPredicate(-21.0, 2, "Y")]
        self.assertEqual(str(exp), "X == 13.0 AND Y >= -21.0")

    def test_predicate_contradictions(self):
        with self.subTest("<< && >>"):
            p1 = LessPredicate(-2.0, 2, "X")
            p2 = GreaterOrEqualPredicate(-2.0, 2, "X")
            self.assertTrue(p1.is_contradictory_to(p2))

        with self.subTest("== && !=="):
            p1 = EqualityPredicate(-2.0, 2, "X")
            p2 = InequalityPredicate(-2.0, 2, "X")
            self.assertTrue(p1.is_contradictory_to(p2))

        with self.subTest("== & == "):
            p1 = EqualityPredicate(-3.0, 2, "X")
            p2 = EqualityPredicate(-2.0, 2, "X")
            self.assertTrue(p1.is_contradictory_to(p2))

        with self.subTest("!= && !="):
            p1 = InequalityPredicate(-2.0, 2, "X")
            p2 = InequalityPredicate(-2.0, 2, "X")
            self.assertFalse(p1.is_contradictory_to(p2))

        with self.subTest(">= && >="):
            p1 = GreaterOrEqualPredicate(-2.0, 2, "Theta")
            p2 = GreaterOrEqualPredicate(-2.0, 2, "Theta")
            self.assertFalse(p1.is_contradictory_to(p2))

        with self.subTest("Different ids"):
            p1 = GreaterOrEqualPredicate(-2.0, 2, "Theta")
            p2 = GreaterOrEqualPredicate(-2.0, 3, "Theta")
            self.assertFalse(p1.is_contradictory_to(p2))

    def test_predicates_simplification(self):
        with self.subTest("Two Less predicates on same feature"):
            exp = TabularExplanation(None, None, feature_values=[[-20, -15, -10, -2, 0, 1, 6, 13, 15], [0, 1]])
            p1 = LessPredicate(13.0, 0, "X")
            p2 = LessPredicate(-2.0, 0, "X")
            exp.predicates = [p1, p2]
            exp.simplify_predicates()
            self.assertEqual(len(exp.predicates), 1)
            self.assertEqual(exp.predicates[0], p2)

        with self.subTest("Two GEQ predicates on same feature"):
            exp = TabularExplanation(None, None, feature_values=[[-20, -15, -10, -2, 0, 1, 6, 13, 15], [0, 1]])
            p1 = GreaterOrEqualPredicate(13.0, 0, "X")
            p2 = GreaterOrEqualPredicate(-2.0, 0, "X")
            exp.predicates = [p1, p2]
            exp.simplify_predicates()
            self.assertEqual(len(exp.predicates), 1)
            self.assertEqual(exp.predicates[0], p1)

        with self.subTest("Two predicates on different features"):
            exp = TabularExplanation(None, None, feature_values=[[-20, -15, -10, -2, 0, 1, 6, 13, 15], [0, 1]])
            p1 = GreaterOrEqualPredicate(13.0, 1, "Y")
            p2 = LessPredicate(-2.0, 0, "X")
            exp.predicates = [p1, p2]
            exp.simplify_predicates()
            self.assertEqual(len(exp.predicates), 2)

