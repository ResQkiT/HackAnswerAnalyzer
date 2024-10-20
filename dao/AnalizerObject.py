import time
from GistogramGenerator import GistogramGenerator

class AnalizerObject:
    def __init__(self, name : str, description : str, analize_function) -> None:
        self.name = name
        self.description = description
        self.analize_function = analize_function
        self.response_object = None;
        self.creation_time = str(time.time_ns()) + str(hash(self))

    def is_valid(self) -> bool:
        return self.response_object != None

    def get_keys_as_list(self) -> list:
        if(not self.is_valid()):
            return []
        
        return list(self.response_object.get_as_map().keys())
    
    def get_values_as_list(self) -> list:
        if(not self.is_valid()):
            return []

        return list(self.response_object.get_as_map().values())
    
    def get_as_pair_list(self) -> list:
        if(not self.is_valid()):
            return []
        
        return list(self.response_object.get_as_map().items())
    
    def get_unique_hash(self):
        return str(hash(self))

    def analize(self, file_path):
        self.response_object = self.analize_function(file_path)
        generator = GistogramGenerator()
        generator.generate_pie_chart(self.response_object.get_as_map(), f"static/images/{self.get_unique_hash()}.png")
