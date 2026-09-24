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


CONTROL_GOALS = {
    "CartPole": "A pole is hinged on a cart; push the cart left or right to keep the pole upright and the cart "
                "on screen.",
    "Acrobot": "Two links hang from a pivot; twist the joint between them to swing the free end up above the "
               "line.",
    "MountainCar": "The car is too weak to drive straight up; rock back and forth to build speed and reach the "
                   "flag on the right hill.",
    "LunarLander": "Fire the engines to land the lander gently and upright between the two flags.",
    "InvertedPendulum": "A pole is hinged on a cart that slides along a rail; push the cart left or right to keep "
                        "the pole upright.",
    "InvertedDoublePendulum": "Two poles are hinged end to end on a cart that slides along a rail; push the cart "
                              "left or right to keep both poles upright.",
    "Reacher": "A two-joint arm lies flat on a table; move its fingertip onto the red target and keep it there "
               "with little effort.",
    "Pusher": "A seven-joint arm stands by a table; push the white cylinder onto the red goal.",
    "Swimmer": "A three-segment swimmer lies in a thick fluid; wriggle its two joints to swim forward.",
    "Hopper": "A one-legged robot stands upright; hop forward as fast as you can without falling.",
    "Walker2d": "A two-legged robot stands upright; walk forward as fast as you can without falling.",
    "HalfCheetah": "A two-legged running robot; run forward as fast as you can.",
    "Ant": "A four-legged robot; walk forward as fast as you can without flipping over.",
    "Humanoid": "A humanoid robot stands upright; walk forward as fast as you can without falling.",
    "HumanoidStandup": "A humanoid robot lies on the ground; get up and raise your head as high as you can.",
}

# MuJoCo joint-torque games: the actuated joints in the environment's actuator order. Each gets a push each way
# (``<JOINT>_POS``, ``<JOINT>_NEG``), after ``NONE``; ``laya.mujocogames`` maps them onto the action vector.
TORQUE_JOINTS = {
    "Reacher": ("shoulder", "elbow"),
    "Pusher": ("shoulder pan", "shoulder lift", "upper arm roll", "elbow flex", "forearm roll", "wrist flex",
               "wrist roll"),
    "Swimmer": ("front joint", "back joint"),
    "Hopper": ("thigh", "knee", "foot"),
    "Walker2d": ("right thigh", "right knee", "right foot", "left thigh", "left knee", "left foot"),
    "HalfCheetah": ("back thigh", "back shin", "back foot", "front thigh", "front shin", "front foot"),
    "Ant": ("leg 4 hip", "leg 4 ankle", "leg 1 hip", "leg 1 ankle", "leg 2 hip", "leg 2 ankle", "leg 3 hip",
            "leg 3 ankle"),
    "Humanoid": ("abdomen y", "abdomen z", "abdomen x", "right hip x", "right hip z", "right hip y", "right knee",
                 "left hip x", "left hip z", "left hip y", "left knee", "right shoulder 1", "right shoulder 2",
                 "right elbow", "left shoulder 1", "left shoulder 2", "left elbow"),
}
TORQUE_JOINTS["HumanoidStandup"] = TORQUE_JOINTS["Humanoid"]


def torque_actions(joints: Sequence[str]) -> Dict[str, str]:
    """``NONE``, then ``<JOINT>_POS`` / ``<JOINT>_NEG`` per joint, with what each does."""
    out = {"NONE": "apply no torque"}
    for j in joints:
        key = j.upper().replace(" ", "_")
        out[key + "_POS"] = "torque the %s one way (positive)" % j
        out[key + "_NEG"] = "torque the %s the other way (negative)" % j
    return out


CONTROL_ACTIONS = {
    "CartPole": {"LEFT": "push the cart left", "RIGHT": "push the cart right"},
    "Acrobot": {"CLOCKWISE": "twist the lower link clockwise", "NONE": "do nothing",
                "COUNTERCLOCKWISE": "twist the lower link counter-clockwise"},
    "MountainCar": {"LEFT": "accelerate left", "NONE": "do not accelerate", "RIGHT": "accelerate right"},
    "LunarLander": {"NOOP": "do nothing", "LEFT_ENGINE": "fire the left orientation engine",
                    "MAIN_ENGINE": "fire the main engine", "RIGHT_ENGINE": "fire the right orientation engine"},
    "InvertedPendulum": {"LEFT": "push the cart left", "NONE": "do not push", "RIGHT": "push the cart right"},
    "InvertedDoublePendulum": {"HARD_LEFT": "push the cart left hard", "LEFT": "push the cart left gently",
                               "NONE": "do not push", "RIGHT": "push the cart right gently",
                               "HARD_RIGHT": "push the cart right hard"},
    **{game: torque_actions(joints) for game, joints in TORQUE_JOINTS.items()},
}


def control_question(game: str) -> Dict:
    """The question for a ``laya.controlgames`` or ``laya.mujocogames`` game; the screen ghosts the previous frame
    to show motion."""
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
