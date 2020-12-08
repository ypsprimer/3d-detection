class Processor(object):
    def __init__(self):
        self.__processes = []

    def add_process(self, process, order=-1):
        if order >= 0:
            self.__processes.insert(order, process)
        else:
            self.__processes.append(process)
        return self.__processes.index(process)

    def run(self):
        for process in self.__processes:
            try:
                process.start()
                process.validate()
            except Exception, e:
                print e.message
                break
