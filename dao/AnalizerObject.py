
class AnalizerObject:
    def __init__(self, name : str, description : str, analize_function) -> None:
        self.name = name
        self.description = description
        self.analize_function = analize_function
        self.response_object = None;

    def get_keys_as_list(self) -> list:
        if(self.response_object == None):
            return []
        
        return list(self.response_object.get_as_map().keys())
    
    def get_values_as_list(self) -> list:
        if(self.response_object == None):
            return []

        return list(self.response_object.get_as_map().values())
    
    def get_as_pair_list(self) -> list:
        if(self.response_object == None):
            return []
        
        return list(self.response_object.get_as_map().items())
    
    def analize(self, file_path):
        self.response_object = self.analize_function(file_path)
