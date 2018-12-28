

class DatasetDescription():

    def __init__(self, description, path, query_id, protected_attribute, protected_attribute_value, protected_group,
                 header, header_in_file, header_to_write, judgment, k, alpha=0.1):
        self.__description = description
        self.__path = path
        self.__query_id = query_id
        self.__protected_attribute = protected_attribute
        self.__protected_attribute_value = protected_attribute_value
        self.__protected_group = protected_group
        self.__header = header
        self.__header_in_file = header_in_file
        self.__header_to_write = header_to_write
        self.__judgment = judgment
        self.__k = k
        self.__alpha = alpha

    @property
    def description(self):
        return self.__description

    @property
    def path(self):
        return self.__path

    @property
    def query_id(self):
        return self.__query_id

    @property
    def protected_attribute(self):
        return self.__protected_attribute

    @property
    def protected_attribute_value(self):
        return self.__protected_attribute_value

    @property
    def protected_group(self):
        return self.__protected_group

    @property
    def header(self):
        return self.__header

    @property
    def header_in_file(self):
        return self.__header_in_file

    @property
    def header_to_write(self):
        return self.__header_to_write

    @property
    def judgment(self):
        return self.__judgment

    @property
    def k(self):
        return self.__k

    @property
    def alpha(self):
        return self.__alpha
