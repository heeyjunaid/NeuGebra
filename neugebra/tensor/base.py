__all__ = ["TensorBase", "Tensor"]


class TensorBase:
    def __init__(self, array, dtype = float):
        self.__tensor = self.__change_dtype(array, dtype)
        self.__len = len(self.__tensor)
        self.dtype = dtype
    
    def __change_dtype(self, items, type_converter = float):
        if isinstance(items, list):
            return [self.__change_dtype(item, type_converter) for item in items]
        return type_converter(items)
    
    def __repr__(self):
        __tensor_str = str(self.__tensor)
        return f"tensor({__tensor_str}, dtype = {self.dtype})"
    
    def __len__(self):
        return self.__len
    
    def __getitem__(self, idx):
        item = self.__tensor[idx]
        if isinstance(item, list):
            return TensorBase(item, self.dtype)
        return item
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx < self.__len:
            item = self.__getitem__(self.idx)
            self.idx += 1
            return item
        else:
            raise StopIteration 
    
    def __recursive_ops(self, array1, array2, ops = "sum"):
        """ function to perform given operation element-wise on tensor 
        """
        if type(array1) in [list, TensorBase] and  type(array2) in [list, TensorBase]:
            return [self.__recursive_ops(a1, a2, ops) for a1, a2 in zip(array1, array2)]
        
        elif isinstance(array1, list) or isinstance(array1, TensorBase):
            return [self.__recursive_ops(a1, array2, ops) for a1 in array1]

        elif isinstance(array2, list) or isinstance(array2, TensorBase):
            return [self.__recursive_ops(a2, array1, ops) for a2 in array2]
        
        if ops == "sum":
            return array1 + array2
        elif ops == "sub":
            return array1 - array2
        elif ops == "div":
            return array1 / array2
        elif ops == "mul":
            return array1 * array2
        elif ops == "mod":
            return array1 % array2
        elif ops == "floor":
            return array1 // array2
        elif ops == "pow":
            return array1**array2
    
    def __add__(self, array):
        res = self.__recursive_ops(self.__tensor, array, "sum")
        return TensorBase(res, self.dtype)
    
    def __sub__(self, array):
        res = self.__recursive_ops(self.__tensor, array, "sub")
        return TensorBase(res, self.dtype)
    
    def __mul__(self, array):
        res = self.__recursive_ops(self.__tensor, array, "mul")
        return TensorBase(res, self.dtype)
    
    def __truediv__(self, array):
        res = self.__recursive_ops(self.__tensor, array, "div")
        return TensorBase(res, self.dtype)
    
    def __floordiv__(self, array):
        res = self.__recursive_ops(self.__tensor, array, "floor")
        return TensorBase(res, self.dtype)
    
    def __pow__(self, array):
        res = self.__recursive_ops(self.__tensor, array, "pow")
        return TensorBase(res, self.dtype)
    
    def __mod__(self, array):
        res = self.__recursive_ops(self.__tensor, array, "mod")
        return TensorBase(res, self.dtype)
    
    def __calculate_shape_recursive(self, arr):
        if isinstance(arr, list):
            return (len(arr), ) + self.__calculate_shape_recursive(arr[0])
        return ()
    
    def items(self):
        """ Returns tensor as list """
        return self.__tensor
    
    def shape(self):
        return self.__calculate_shape_recursive(self.__tensor)
    
    def astype(self, dtype):
        self.dtype = dtype
        self.__tensor = self.__change_dtype(self.__tensor, dtype)
        return self


class Tensor(TensorBase):
    def __init__(self, array, dtype = float) -> None:
        super().__init__(array, dtype)

    def transpose(self):
        """ Function to transpose the tensor
        """
        pass
    
    def inverse(self):
        pass
    
    def dot(self, array2):
        pass
    
    def cross(self, array2):
        pass

