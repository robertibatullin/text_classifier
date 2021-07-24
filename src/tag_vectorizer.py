from typing import List

import pandas as pd

class TagVectorizer:

    def vectorize(self, tag_id_lines: List[List[int]]) -> pd.DataFrame:
        """
        [[0,1], [1]] --> pd.DataFrame({0: [1,0], 1: [1,1]})
        """
        records = []
        max_tag_id = 0
        for tag_id_line in tag_id_lines:
            record = {tag_id: 1 for tag_id in set(tag_id_line)
                     if tag_id >= 0}
            max_tag_id = max([max_tag_id]+tag_id_line)
            records.append(record)
        df = pd.DataFrame.from_records(records)
        df = df.reindex(range(max_tag_id+1), axis=1)
        df.fillna(0, inplace=True)
        df = df.applymap(int)
        return df
