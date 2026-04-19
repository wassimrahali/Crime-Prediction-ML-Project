import unittest

import numpy as np
import pandas as pd

from src.preprocessing import RAW_TARGET_COL, clean_raw, make_target


class TestPreprocessing(unittest.TestCase):
	def test_make_target_yes_no(self):
		df = pd.DataFrame({RAW_TARGET_COL: ["Yes", "No", " yes ", "NO", "Unknown", None]})
		y = make_target(df)
		self.assertEqual(list(y[:4].astype("Int64")), [1, 0, 1, 0])
		self.assertTrue(pd.isna(y.iloc[4]))
		self.assertTrue(pd.isna(y.iloc[5]))

	def test_make_target_numeric_strings(self):
		df = pd.DataFrame({RAW_TARGET_COL: ["1", "0", "1", None]})
		y = make_target(df)
		self.assertEqual(list(y[:3].astype("Int64")), [1, 0, 1])
		self.assertTrue(pd.isna(y.iloc[3]))

	def test_clean_raw_perp_age_zero_to_nan(self):
		df = pd.DataFrame({"Perpetrator Age": [0, 5, "0", "10"]})
		out = clean_raw(df)
		self.assertTrue(np.isnan(out.loc[0, "Perpetrator Age"]))
		self.assertEqual(out.loc[1, "Perpetrator Age"], 5)
		self.assertTrue(np.isnan(out.loc[2, "Perpetrator Age"]))
		self.assertEqual(out.loc[3, "Perpetrator Age"], 10)


if __name__ == "__main__":
	unittest.main()
