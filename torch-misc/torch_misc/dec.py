from functools import wraps


class CachedProperty(property):
    """Subclass used to signal cached properties."""
    pass


def cache(method):
    """Decorator to cache selected property values.
    
    Can use del obj.property to clear the cache."""
    name = '_' + method.__name__
        
    @CachedProperty
    @wraps(method)
    def f(self):
        if hasattr(self, name):
            res = getattr(self, name)
        else:
            res = method(self)
            object.__setattr__(self, name, res)
            
        return res
        
    @f.deleter
    def f(self):
        if hasattr(self, name):
            delattr(self, name)

    return f


def cache_only_in_eval(cls):
    """Decorator for torch.nn.Module to make any `CachedProperty` 
    clear its cache whenever the Module is in training mode. 
    The result of the property will be cached only in eval mode.
    """
    
    # Identify all cached properties
    cached_properties = [
        name
        for name in dir(cls)
        if isinstance(getattr(cls, name), CachedProperty)
    ]
    
    # Decorate all cached properties again so that, if they're in training, 
    # the cached value disappears after access
    def dec(method):
        
        @wraps(method)
        def f(self):
            res = method(self)
            
            if self.training:
                delattr(self, method.__name__)
            
            return res
        
        return f
    
    for name in cached_properties:
        p = getattr(cls, name)
        setattr(cls, name, wraps(p.fget)(CachedProperty(dec(p.fget), fdel=p.fdel)))
        
    # Also, edit the train method so that when coming from eval,
    # we also destroy the cached values    
    def train(self, train=True):
        # Delete all caches
        for name in cached_properties:
            delattr(self, name)
                
        return super(cls, self).train(train)
    
    cls.train = train
    return cls