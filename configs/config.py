import os
import yaml


class Config(object):
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def load(cls, file_path):
        with open(file_path,'r',encoding="utf-8") as stream:
            data = yaml.load(stream)
        for (k, v) in data.items():
            setattr(cls, k, v)
        return cls

    @classmethod
    def init(cls):
        cls.load(os.path.join(cls.PROJECT_PATH,'test.yml'))
        return cls


    @classmethod
    def update(cls, file_path):
         with open(file_path,'r',encoding="utf-8") as stream:
             data = yaml.load(stream)
         for (k, v) in data.items():
             if not hasattr(cls,k):
                 setattr(cls, k, v)
             else:
                 if isinstance(v, dict):
                     tmp = getattr(cls, k)
                     for k1,v1 in v.items():
                         tmp[k1] = v1
                     setattr(cls, k, tmp)
                 else:
                     setattr(cls, k, v)


if __name__ == "__main__":
    Config.load("/yupeng/alg-coronary-seg3d/configs/test.yml")
    # print(vars(Config))
    print(Config.net)
