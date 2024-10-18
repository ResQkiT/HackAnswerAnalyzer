class ResponseObject:   
    def __init__(self, data_map : map) -> None:
        self.response_map = data_map
        
    def get_as_map(self) -> map:
        return self.response_map

        
    
