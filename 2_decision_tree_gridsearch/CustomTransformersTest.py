import pandas as pd
import numpy as np
import unittest
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.util.testing import assert_frame_equal

from CustomTransformers import (SelectColumns, FillNaCommon, FillNaWithConst, ProcessCategoriesAsIndex, 
                                ProcessCategoriesOHE, ProcessBins)


class TestSelectColumns(unittest.TestCase):

    def test_transform(self):
        test_df = pd.util.testing.makeMixedDataFrame()
        baseline = list(test_df.columns)
        columns = ['A', 'B']

        transformer = SelectColumns(columns)
        processed_data = transformer.fit_transform(test_df)

        self.assertListEqual(list(processed_data.columns), columns, "Should extract only specified")
        self.assertListEqual(list(test_df.columns), baseline, "Should not change input")

        
class TestFillNaCommon(unittest.TestCase):

    def test_transform(self):
        test_df = pd.util.testing.makeMixedDataFrame()
        column = 'A'
        replacement = test_df[column].mode()
        
        test_df.loc[test_df['A'] == 0, 'A'] = None
        
        transformer = FillNaCommon([column])
        processed_data = transformer.fit_transform(test_df)
        
        self.assertEqual(processed_data[column].isna().sum(), 0, "Should fill Na s")
        self.assertEqual(test_df[column].isna().sum(), 1, "Should not change input")
    
    def test_transform_string_column(self):
        test_df = pd.util.testing.makeMixedDataFrame()
        column = 'C'   
        
        test_df.at[0, column] = 'foo2'
        test_df.at[1, column] = 'foo2'
        test_df[column].iat[-1] = None
        
        replacement = test_df[column].mode()
        
        transformer = FillNaCommon([column])
        processed_data = transformer.fit_transform(test_df)
        
        self.assertEqual(processed_data[column].isna().sum(), 0, "Should fill Na s")
        self.assertEqual(processed_data[column].iat[-1], 'foo2', "Should fill Na with mode")
        self.assertEqual(test_df[column].isna().sum(), 1, "Should not change input")


class TestFillNaWithConst(unittest.TestCase):

    def test_transform(self):
        test_df = pd.util.testing.makeMixedDataFrame()
        column = 'A'
        replacement = 'batman'
        
        test_df.at[0, column] = None
        test_df[column].iat[-1] = None
        
        transformer = FillNaWithConst(replacement, [column])
        processed_data = transformer.fit_transform(test_df)
        count_replacement = len(processed_data[processed_data[column] == replacement])
        
        self.assertEqual(processed_data[column].isna().sum(), 0, "Should fill Na s")
        self.assertEqual(count_replacement, 2, "Should fill Na with replacement")
        self.assertEqual(test_df[column].isna().sum(), 2, "Should not change input")
        
        
class TestProcessCategoriesAsIndex(unittest.TestCase):

    def test_transform(self):
        test_df = pd.util.testing.makeMixedDataFrame()
        column = 'C'
        
        transformer = ProcessCategoriesAsIndex([column])
        processed_data = transformer.fit_transform(test_df)
        non_numeric = len(pd.to_numeric(processed_data[column], errors='coerce'))

        self.assertEqual(non_numeric, len(processed_data[column]), "Should transform category to index")
        self.assertTrue(is_numeric_dtype(processed_data[column]), "Should transform {} to numeric".format(column))
        self.assertTrue(is_string_dtype(test_df[column]), "Should not change input")
        
        
class TestProcessCategoriesOHE(unittest.TestCase):

    def test_transform(self):
        test_df = pd.util.testing.makeMixedDataFrame()
        column = 'C'
        
        transformer = ProcessCategoriesOHE([column])
        processed_data = transformer.fit_transform(test_df)
        num_columns = len(processed_data.columns)
        category_columns = [col for col in processed_data if col.startswith(column)]
        
        self.assertEqual(num_columns, 8, "Should transform category to ohe")
        for col in category_columns:
            self.assertTrue(is_numeric_dtype(processed_data[col]), "Should transform {} to numeric ".format(col))
        self.assertTrue(is_string_dtype(test_df[column]), "Should not change input")
        

class TestProcessBins(unittest.TestCase):

    def test_transform(self):
        test_df = pd.util.testing.makeMixedDataFrame()
        column = 'A'
        bins = [0,2,4,6]
        labels = ['S', 'M', 'L']
        
        transformer = ProcessBins({'A': 
                                   {'bins': bins, 
                                    'labels': labels}
                                  })
        processed_data = transformer.fit_transform(test_df)
        num_columns = len(processed_data.columns)
        category_columns = [col for col in processed_data if col.startswith(column)]
        
        self.assertEqual(num_columns, 6, "Should transform category to ohe")
        for col in category_columns:
            self.assertTrue(is_numeric_dtype(processed_data[col]), "Should transform {} to numeric ".format(col))
        self.assertEqual(len(test_df.columns), 4, "SShould not change input columns number")
        self.assertTrue(is_numeric_dtype(test_df[column]), "Should not change input")
        
    def test_transform_empty(self):
        test_df = pd.util.testing.makeMixedDataFrame()
        
        transformer = ProcessBins()
        processed_data = transformer.fit_transform(test_df)
        
        assert_frame_equal(processed_data, test_df, "Should not transform with empty columns")
        
        
if __name__ == '__main__':
    unittest.main()