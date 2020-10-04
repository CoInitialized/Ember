import re
import pandas as pd

class NameFixer:

    def __init__(self):
        self.fixed_columns = {}

    @staticmethod
    def fix_forbidden_name(x):
        error_regex = re.compile(r"([<>\[\]])")
        result =  re.sub(error_regex, '', x)
        if result == x:
            return None
        else:
            return result

    def fit(self, X, y = None):

        if isinstance(X, pd.DataFrame):

            for i, column in enumerate(X.columns):
                fixed = self.fix_forbidden_name(column)
                if fixed:
                    self.fixed_columns[column] = fixed
        else:
            pass

        return self

    def transform(self, X, y = None):

        if self.fixed_columns:
            print(self.fixed_columns)
            return X
        else:
            return X.rename(columns = self.fixed_columns)


