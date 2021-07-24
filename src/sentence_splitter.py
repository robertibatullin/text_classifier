import re
from typing import List


class SentenceSplitter:
    
    SENTENCE_SEPARATOR_PATTERN = '[\.\?\!…;:]| - |\n ?["\-–«]|, ?[\-–]|--'
    MARKS = '.,:;?!(){}[]|"\n_…«»–-'
    
    def split(self, text: str) -> List[str]:
        sentences = re.split(self.SENTENCE_SEPARATOR_PATTERN, text)
        separators = re.findall(self.SENTENCE_SEPARATOR_PATTERN, text)
        split = [sentence.strip(self.MARKS).replace('\n',' ').strip()+separator.strip()
                 for sentence, separator in zip(sentences, separators)
                 if len(sentence.strip(self.MARKS+' '))>0]
        return split
