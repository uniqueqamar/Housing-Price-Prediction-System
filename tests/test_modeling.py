import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from modeling import FEATURES, load_data, train_and_evaluate


class EvaluationTests(unittest.TestCase):
    def test_holdout_and_baseline(self):
        df = load_data()
        model, report, actual, predicted = train_and_evaluate(df)
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        self.assertTrue(set(train.index).isdisjoint(test.index))
        self.assertEqual(list(actual.index), list(test.index))
        self.assertEqual(report['test_rows'], 109)
        np.testing.assert_array_equal(model.named_steps['preprocessing'].transformers_[1][1].categories_[2],
                                   sorted(train.furnishingstatus.unique()))
        expected = np.mean(np.abs(test.price - train.price.mean()))
        self.assertAlmostEqual(report['mean_baseline']['mae_inr'], expected)
        np.testing.assert_allclose(model.predict(test[FEATURES]), predicted)
        self.assertTrue(np.isfinite(predicted).all())

    def test_invalid_categories_are_rejected(self):
        df = load_data()
        model, _, _, _ = train_and_evaluate(df)
        row = df[FEATURES].iloc[:1].copy()
        row['furnishingstatus'] = 'unknown'
        with self.assertRaises(ValueError):
            model.predict(row)


if __name__ == '__main__':
    unittest.main()
