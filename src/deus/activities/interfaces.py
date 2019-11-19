import abc


class Observer:
    @abc.abstractmethod
    def update(self):
        raise NotImplementedError


class Subject:
    @abc.abstractmethod
    def attach(self, o):
        raise NotImplementedError

    @abc.abstractmethod
    def detach(self, o):
        raise NotImplementedError

    @abc.abstractmethod
    def notify_observers(self):
        raise NotImplementedError
