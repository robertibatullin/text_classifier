from typing import List
from collections import Counter


class TagMapper:
    UNKNOWN_TAG = 'unknown'
    UNKNOWN_TAG_ID = -1

    def __init__(self,
                 tags: List[str]):
        self.__tags = tags
        
    @classmethod
    def from_taglines(cls, 
                      taglines: List[str]):
        """
        :param taglines: list of strings, each string containing
        comma-separated tags
        """
        tags = [tag.lower().strip() for tagline in taglines
                for tag in tagline.split(',')]
        counter = Counter(tags)
        # filtering out tags that occur only once
        filtered = filter(lambda tag: counter[tag] > 1, set(tags))
        tags = sorted(filtered)
        return cls(tags)

    @property
    def tags(self) -> List[str]:
        return self.__tags

    def tag_to_id(self, tag: str) -> int:
        if tag in self.__tags:
            return self.__tags.index(tag)
        return self.UNKNOWN_TAG_ID

    def id_to_tag(self, id_: int) -> str:
        if id_ in range(len(self.__tags)):
            return self.__tags[id_]
        return self.UNKNOWN_TAG

    def tagline_to_ids(self, tagline: str) -> List[int]:
        """
        :param tagline: string of comma-separated tags
        :return: list of corresponding ids
        """
        split = [tag.lower().strip() for tag in tagline.split(',')]
        return list(map(self.tag_to_id, split))

    def ids_to_taglists(self, ids: List[int]) -> List[str]:
        """
        :param ids: list of ids
        :return: string of comma-separated corresponding tags
        """
        return list(map(self.id_to_tag, ids))
