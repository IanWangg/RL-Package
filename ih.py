class father(object):
    def __init__(self):
        pass

    def yell(self):
        print('father : Hello!')

    def start_yell(self):
        self.yell()


class son(father):
    def __init__(self):
        pass
    
    def yell(self):
        print('son : Hello!')

    def start_yell(self):
        super().start_yell()

if __name__ == '__main__':
    son1 = son()
    print(son1.start_yell())
