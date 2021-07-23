from typing import List

import pandas as pd

class TagVectorizer:

    def vectorize(self, tag_id_lines: List[List[int]]) -> pd.DataFrame:
        """
        [[0,1], [1]] --> pd.DataFrame({0: [1,0], 1: [1,1]})
        """
        records = []
        for tag_id_line in tag_id_lines:
            record = {tag_id: 1 for tag_id in set(tag_id_line)}
            records.append(record)
        df = pd.DataFrame.from_records(records)
        df.fillna(0, inplace=True)
        df.sort_index(axis=1, inplace=True)
        df = df.applymap(int)
        return df
