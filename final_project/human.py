#!/home/dylan/.virtualenvs/superGo/bin/python

import click
from lib.play import play, Game
from utils.utils import load_player
from lib.gtp import Engine


@click.command()
@click.option("--folder", default=-1)
@click.option("--ite", default=-1)
@click.option("--gtp/--no-gtp", default=False)
def main(round, ite, gtp):
    player, _ = load_player(round)
    if not isinstance(player, str):
        game = Game(player, 0)
        engine = Engine(game, board_size=game.gobang_size)
        while True:
            print(engine.send(input()))
    elif not gtp:
        print(player)
    else:
        print("¯\_(ツ)_/¯")


if __name__ == "__main__":
    main()