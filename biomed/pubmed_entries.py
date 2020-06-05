class PubmedEntry:
    def __init__(self, pmid: int, text: str):
        self.__pmid = pmid
        self.text = text

    def get_pmid(self):
        return self.__pmid

    pmid = property(get_pmid)


class PubmedTrainingEntry(PubmedEntry):
    def __init__(self, pmid: int, text: str, is_cancer: int):
        super(PubmedTrainingEntry, self).__init__(pmid, text)
        self.__is_cancer = is_cancer

    def get_is_cancer(self):
        return self.__is_cancer

    is_cancer = property(get_is_cancer)
