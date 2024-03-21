from argparse import ArgumentParser

from loader import Loader
from model import Model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("exam", help="examination recording history")
    parser.add_argument("eco", help="edu-eco")
    parser.add_argument("tele_history", help="inviting history")
    parser.add_argument("list", help="inviting list")
    args = vars(parser.parse_args())

    loader = Loader()
    if loader.read_record(exam=args["exam"], eco=args["eco"], tele_history=args["tele_history"], tele_pred=args["list"]) != 0:
        print("資料缺漏，停止執行")
        exit()

    loader.preprocessing()
    X, y, train_area = loader.to_training_set()

    mod = Model()
    mod.train(X, y)

    pX, first, no_screen, regular, useless, reject, addr = loader.to_predicting_set(train_area)

    result = mod.pred(pX)
    loader.save(result, first, no_screen, regular, useless, reject, addr)
