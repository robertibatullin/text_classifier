from typing import List, Set
import re

from .sentence_splitter import SentenceSplitter


class ProperNamesFinder:
    
    INITIAL_CAPITALS = 'ЁЙЦУКЕНГШЩЗХФВАПРОЛДЖЭЯЧСМИТБЮQWERTYUIOPASDFGHJKLZXCVBNM'
    
    def __init__(self):
        self.__proper_names = set()
        self.__not_proper_names = set()
        self.__splitter = SentenceSplitter()
        
    @property
    def proper_names(self) -> Set[str]:
        return self.__proper_names
    
    @property
    def not_proper_names(self) -> Set[str]:
        return self.__not_proper_names

    def reset(self):
        self.__proper_names = set()
        self.__not_proper_names = set()
        
    def __get_lowercase_words(self, text: str) -> Set[str]:
        words = set(re.findall(r'\b\w+\b', text))
        lowercase_words = [word for word in words
                          if word.lower() == word]
        return set(lowercase_words)
    
    def __get_capitalized_words(self, text: str) -> Set[str]:
        # words non-first in sentences and starting with capital letter
        cap_words = set()
        sentences = self.__splitter.split(text)
        for sent in sentences:
            sent_words = re.split('[ \n\t]+',sent)
            if len(sent_words)<2:
                continue
            stripped_sent_words = []
            for i,word in enumerate(sent_words):
                stripped = word.strip(self.__splitter.MARKS+' ')
                if len(stripped) > 1:
                    if stripped[0] in self.INITIAL_CAPITALS and i>0:
                        cap_words.add(stripped)
        return cap_words
    
    def fit(self, text: str):
        lowercase_words = self.__get_lowercase_words(text)
        cap_words = self.__get_capitalized_words(text)
        self.__not_proper_names.update(lowercase_words)
        self.__proper_names.update([word for word in cap_words
                                   if word.lower() not in self.__not_proper_names])
