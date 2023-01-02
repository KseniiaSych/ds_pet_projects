import pandas as pd
import numpy as np
import unittest
from pandas.util.testing import assert_frame_equal

from TitanicSpecificTransformers import (ExctractTitle, ExctractDeck, FamilySize, FeatureIsAlone)
        
        
class TestExctractTitle(unittest.TestCase):

    def test_transform(self):
        lst = ['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina', 'Allen, Mr. William Henry']
        test_df = pd.DataFrame(lst, columns =['Name'])
        
        transformer = ExctractTitle()
        processed_data = transformer.fit_transform(test_df)
        
        self.assertTrue('Title' in processed_data.columns, "Should add title columns")
        self.assertEqual(processed_data['Title'].iloc[0], 'Mr', "Should exctract title")
        self.assertEqual(processed_data['Title'].iloc[1], 'Miss', "Should exctract title")
        self.assertEqual(processed_data['Title'].iloc[2], 'Mr', "Should exctract title")

        
class TestExctractDeck(unittest.TestCase):

    def test_transform(self):
        lst = ['C123', 'D85', None]
        test_df = pd.DataFrame(lst, columns =['Cabin'])

        transformer = ExctractDeck()
        processed_data = transformer.fit_transform(test_df)
        
        self.assertTrue('Deck' in processed_data.columns, "Should add deck columns")
        self.assertEqual(processed_data['Deck'].iloc[0], "ABC", "Should exctract deck")
        self.assertEqual(processed_data['Deck'].iloc[1], "DE", "Should exctract deck")
        self.assertEqual(processed_data['Deck'].iloc[2], "M", "Should exctract missing")
        
        
class TestFamilySize(unittest.TestCase):

    def test_transform(self):
        sib = [1, None]
        parch = [3, 0]
        test_df = pd.DataFrame(list(zip(sib, parch)), columns =['SibSp', 'Parch'])

        transformer = FamilySize()
        processed_data = transformer.fit_transform(test_df)
        
        self.assertTrue('FamilySize' in processed_data.columns, "Should add familySize columns")
        self.assertEqual(processed_data['FamilySize'].iloc[0], 5, "Should exctract familySize")
        self.assertEqual(processed_data['FamilySize'].iloc[1], 1, "Should exctract alone passager")
        
        
class TestFeatureIsAlone(unittest.TestCase):

    def test_transform(self):
        sib = [1, None]
        parch = [3, 0]
        test_df = pd.DataFrame(list(zip(sib, parch)), columns =['SibSp', 'Parch'])

        transformer = FeatureIsAlone()
        processed_data = transformer.fit_transform(test_df)
        print(processed_data.columns)
        
        self.assertFalse('tmp_FamilySize' in processed_data.columns, "Should not add familySize columns")
        self.assertTrue('IsAlone' in processed_data.columns, "Should add IsAlone columns")
        self.assertEqual(processed_data['IsAlone'].iloc[0], 0, "Should exctract isAlone")
        self.assertEqual(processed_data['IsAlone'].iloc[1], 1, "Should exctract isAlone")
        
        
if __name__ == '__main__':
    unittest.main()