"""Question builders for playing games with the image model, shared by training-data generation and the live
viewers (``examples/atari_live.py``, ``examples/vizdoom_live.py``) so both ask exactly the same question.

Each game step is one ``choice`` question: the screen is the image, the options are the game's actions. Atari
and ViZDoom run in their emulators; Maze and Snake are ``laya.gridgames``.
"""
from typing import Dict, List, Optional, Sequence

ATARI_ACTIONS = {
    "NOOP": "do nothing", "FIRE": "press fire (launch the ball / shoot)", "UP": "move up", "DOWN": "move down",
    "RIGHT": "move right", "LEFT": "move left", "RIGHTFIRE": "move right and fire", "LEFTFIRE": "move left and fire",
    "UPFIRE": "move up and fire", "DOWNFIRE": "move down and fire",
    "UPRIGHT": "move up and right", "UPLEFT": "move up and left", "DOWNRIGHT": "move down and right",
    "DOWNLEFT": "move down and left", "UPRIGHTFIRE": "move up-right and fire", "UPLEFTFIRE": "move up-left and fire",
    "DOWNRIGHTFIRE": "move down-right and fire", "DOWNLEFTFIRE": "move down-left and fire",
}
ATARI_GOALS = {
    "Breakout": "Keep the ball in play with the paddle at the bottom and break the bricks.",
    "Pong": "You control the right paddle. Hit the ball back past the opponent on the left.",
    "Freeway": "You control the chicken on the left. Move it up across the road to the top while avoiding the cars.",
    "SpaceInvaders": "You control the cannon at the bottom. Shoot the aliens above and dodge their shots.",
    "Skiing": "Steer the skier down the slope, passing between each pair of flags.",
    "Boxing": "You are the white boxer. Get close to the black boxer and punch him.",
    "MsPacman": "Guide Ms. Pac-Man through the maze to eat the dots while avoiding the ghosts.",
    "Pacman": "Guide Pac-Man through the maze to eat the dots while avoiding the ghosts.",
    "Enduro": "Drive the car forward and pass the other cars without crashing.",
}

DOOM_BUTTONS = {
    "MOVE_LEFT": "strafe left", "MOVE_RIGHT": "strafe right", "MOVE_FORWARD": "move forward",
    "MOVE_BACKWARD": "move backward", "TURN_LEFT": "turn left", "TURN_RIGHT": "turn right", "ATTACK": "shoot",
}
DOOM_GOALS = {
    "basic": "A monster is in the room. Line up with it by strafing left or right, and shoot when it is in the "
             "center of your view.",
    "defend_the_center": "You stand in the middle of a circular room. Turn to face the monsters and shoot them "
                         "before they reach you.",
    "defend_the_line": "Monsters come at you from the far wall. Turn to face them and shoot them.",
    "health_gathering": "The floor hurts you. Walk around and pick up the green medikits to stay alive.",
    "take_cover": "Monsters shoot fireballs at you. Strafe left or right to dodge them.",
    "predict_position": "Aim ahead of the moving monster and fire a rocket so it hits.",
    "deadly_corridor": "Move forward down the corridor to the armor at the end, shooting the monsters on the sides.",
    "my_way_home": "Find your way through the rooms to the green armor.",
}


def atari_question(game: str, actions: Sequence[str]) -> Dict:
    return {"action": {
        "type": "choice",
        "instructions": "You are playing the Atari game %s. %s Which action should the player take now?"
                        % (game, ATARI_GOALS.get(game, "Score as many points as possible.")),
        "criteria": {a: ATARI_ACTIONS.get(a, a.lower()) for a in actions},
    }}


GRID_MOVES = {"UP": "move up", "DOWN": "move down", "LEFT": "move left", "RIGHT": "move right"}


def maze_question() -> Dict:
    """The question for ``laya.gridgames.Maze``: the blue square walks the white corridors to the green one."""
    return {"action": {
        "type": "choice",
        "instructions": "You are the blue square in a maze. Black cells are walls; move along the white corridors "
                        "to reach the green square. Which way should you move now?",
        "criteria": dict(GRID_MOVES),
    }}


def snake_question() -> Dict:
    """The question for ``laya.gridgames.Snake``: the dark green head leads the light green body."""
    return {"action": {
        "type": "choice",
        "instructions": "You are playing Snake. The dark green square is the snake's head and the light green "
                        "squares are its body. Eat the red food, and do not run into the black walls or your own "
                        "body. Which way should the snake move now?",
        "criteria": dict(GRID_MOVES),
    }}


PAINT_DIRECTIONS = {
    "N": "straight up", "NNE": "up and slightly right", "NE": "diagonally up and right",
    "ENE": "right and slightly up", "E": "straight right", "ESE": "right and slightly down",
    "SE": "diagonally down and right", "SSE": "down and slightly right", "S": "straight down",
    "SSW": "down and slightly left", "SW": "diagonally down and left", "WSW": "left and slightly down",
    "W": "straight left", "WNW": "left and slightly up", "NW": "diagonally up and left", "NNW": "up and slightly left",
}
PAINT_PEN = {
    "PEN_DOWN": "press the mouse button to start drawing", "PEN_UP": "release the mouse button to stop drawing",
    "DONE": "the drawing is finished",
}
PAINT_GOALS = {"circle": "draw one round, closed circle, about as big as a third of the canvas height",
               "square": "draw one closed square with straight sides, about a third of the canvas across"}


def paint_goal(task: str) -> str:
    """The task sentence for a drawing task: the hand-written goals for ``circle`` / ``square``, otherwise a simple
    doodle of the named thing (any Quick, Draw! category, e.g. ``house`` or ``smiley face``)."""
    if task in PAINT_GOALS:
        return PAINT_GOALS[task]
    article = "an" if task[:1].lower() in "aeiou" else "a"
    return "draw %s %s as a simple line doodle, about two fifths of the canvas across" % (article, task)


def paint_question(task: str = "circle", directions: int = 32, step_px: int = 6) -> Dict:
    """The question for ``laya.paintenv.JSPaintEnv``: a paint canvas, a red cursor, and mouse-only actions (a
    ``step_px`` move toward each of ``directions`` compass points, then pen down / pen up / done)."""
    from laya.paintenv import compass_bearing, compass_moves

    moves = {m: "move the cursor %d pixels at bearing %g degrees (clockwise from straight up)%s" % (
        step_px, compass_bearing(m), ", " + PAINT_DIRECTIONS[m] if m in PAINT_DIRECTIONS else "")
        for m in compass_moves(directions)}
    return {"action": {
        "type": "choice",
        "instructions": "You are using a paint program with only the mouse. You see the canvas a few steps ago "
                        "and now. The white area is the canvas and the red mark is the mouse cursor: a hollow ring "
                        "with a cross means the button is up, a filled dot means it is held down and moving draws a "
                        "black line. Your task: %s. Which mouse action should you take now?" % paint_goal(task),
        "criteria": {**moves, **PAINT_PEN},
    }}


PAINT_SHAPES = {
    "circle": "a round, closed circle", "oval": "an oval or ellipse, round but stretched",
    "arc": "a curved line or an unclosed part of a circle", "line": "one or more straight lines",
    "square": "a square or rectangle", "triangle": "a triangle", "scribble": "random scribbles",
    "nothing": "nothing, the canvas is blank",
}


def paint_judgements(task: str = "circle", shapes: Optional[Dict[str, str]] = None) -> Dict:
    """The judgement questions asked with ``paint_question`` each step (same state, same ``predict`` call):

    * ``progress`` (``score``, 0-4): how far the canvas is toward the task, the model's own dense signal;
    * ``on_track`` (``choice``): whether the drawing so far is heading toward the task. Named options, because on
      line drawings the model's yes/no (``noul``) answers were unreliable while named choices were not;
    * ``drawn`` (``choice`` over ``PAINT_SHAPES``): what the canvas shows now, whatever the task. At the end of an
      episode this labels what was actually drawn, so a failed attempt can be relabelled as an example of that.
      ``shapes`` replaces ``PAINT_SHAPES`` as its options (e.g. the Quick, Draw! training categories plus
      ``nothing``).
    """
    goal = paint_goal(task)
    return {
        "progress": {
            "type": "score",
            "instructions": "The task is to %s. Looking at the canvas now, how far along is the drawing?" % goal,
            "criteria": ["nothing useful drawn yet", "started, but far from done", "about half done",
                         "nearly done", "the task is complete"],
        },
        "on_track": {
            "type": "choice",
            "instructions": "The task is to %s. Is the drawing so far heading toward that?" % goal,
            "criteria": {"on track": "what is drawn so far could become the task by continuing",
                         "off track": "what is drawn so far will not become the task"},
        },
        "drawn": {
            "type": "choice",
            "instructions": "What is drawn on this white canvas?",
            "criteria": dict(shapes or PAINT_SHAPES),
        },
    }


CONTROL_GOALS = {
    "CartPole": "A pole is hinged on a cart; push the cart left or right to keep the pole upright and the cart "
                "on screen.",
    "Acrobot": "Two links hang from a pivot; twist the joint between them to swing the free end up above the "
               "line.",
    "MountainCar": "The car is too weak to drive straight up; rock back and forth to build speed and reach the "
                   "flag on the right hill.",
    "LunarLander": "Fire the engines to land the lander gently and upright between the two flags.",
}
CONTROL_ACTIONS = {
    "CartPole": {"LEFT": "push the cart left", "RIGHT": "push the cart right"},
    "Acrobot": {"CLOCKWISE": "twist the lower link clockwise", "NONE": "do nothing",
                "COUNTERCLOCKWISE": "twist the lower link counter-clockwise"},
    "MountainCar": {"LEFT": "accelerate left", "NONE": "do not accelerate", "RIGHT": "accelerate right"},
    "LunarLander": {"NOOP": "do nothing", "LEFT_ENGINE": "fire the left orientation engine",
                    "MAIN_ENGINE": "fire the main engine", "RIGHT_ENGINE": "fire the right orientation engine"},
}


def control_question(game: str) -> Dict:
    """The question for a ``laya.controlgames`` game; the screen ghosts the previous frame to show motion."""
    return {"action": {
        "type": "choice",
        "instructions": "You are playing the control task %s. %s A faint copy shows where things were one step "
                        "earlier. Which action should you take now?" % (game, CONTROL_GOALS[game]),
        "criteria": dict(CONTROL_ACTIONS[game]),
    }}


def doom_question(scenario: str, buttons: Sequence[str]) -> Dict:
    return {"action": {
        "type": "choice",
        "instructions": "You are playing Doom. %s Which action should the player take now?"
                        % DOOM_GOALS.get(scenario, "Survive and complete the level."),
        "criteria": {b: DOOM_BUTTONS.get(b, b.lower().replace("_", " ")) for b in buttons},
    }}


_NOT_MONSTERS = {"DoomPlayer", "BulletPuff", "Blood"}


def doom_basic_expert(labels, screen_width: int = 320, margin: int = 2) -> Optional[str]:
    """Scripted expert for ViZDoom ``basic`` from the labels buffer (object bounding boxes in screen pixels).

    ATTACK if the monster's box covers the crosshair column (with ``margin`` pixels to spare), otherwise strafe
    toward it. Returns None when no monster is visible.
    """
    mons = [l for l in labels if l.object_name not in _NOT_MONSTERS and not l.object_name.startswith("Dead")]
    if not mons:
        return None
    m = max(mons, key=lambda l: l.width * l.height)
    cx = screen_width / 2
    if m.x + margin <= cx <= m.x + m.width - margin:
        return "ATTACK"
    return "MOVE_LEFT" if m.x + m.width / 2 < cx else "MOVE_RIGHT"


def doom_buttons(game) -> List[str]:
    return [str(b).split(".")[-1] for b in game.get_available_buttons()]
