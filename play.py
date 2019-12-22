import os
import random
import configparser
import gc
import logging
import re
import yaml
from pathlib import Path
from random import shuffle
from shutil import get_terminal_size

from generator.gpt2.gpt2_generator import GPT2Generator
from story.story_manager import (
    StoryManager,
    UnconstrainedStoryManager,
    ConstrainedStoryManager,
)
from story.utils import logger, player_died, player_won, first_to_second_person, get_similarity, cut_trailing_sentence, standardize_punctuation
import textwrap
import transformers.tokenization_utils

# silence transformers outputs when loading model
logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARN)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)

# add color for windows users that install colorama
try:
    import colorama

    colorama.init()
except ModuleNotFoundError:
    pass

with open(Path("interface", "clover"), "r", encoding="utf-8") as file:
    print(file.read())

# perhaps all the following should be put in a seperate utils file like original
config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
settings = config["settings"]
config_act = settings["actions"]
colors = config["colors"]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.setLevel(settings["log-level"])


def colPrint(str, col="0", wrap=True):
    # ECMA-48 set graphics codes for the curious. Check out "man console_codes"
    if wrap and settings["text-wrap-width"] > 1:
        str = textwrap.fill(str, settings["text-wrap-width"], replace_whitespace=False)
    print("\x1B[{}m{}\x1B[{}m".format(col, str, colors["default"]))


def colInput(str, col1=colors["default"], col2=colors["default"]):
    val = input("\x1B[{}m{}\x1B[0m\x1B[{}m".format(col1, str, col1))
    print("\x1B[0m", end="")
    return val


def getNumberInput(n):
    while True:
        val = colInput(
            "Enter a number from above (default 0):",
            colors["selection-prompt"],
            colors["selection-value"],
        )
        if val == "":
            return 0
        try:
            val = int(val)
        except ValueError:
            colPrint("Invalid number.", colors["error"])
            continue
        if 0 > val or val > n:
            colPrint("Invalid choice.", colors["error"])
            continue
        else:
            return val


def selectFile(p=Path("prompts")):
    if p.is_dir():
        files = [x for x in p.iterdir()]
        shuffle(files)
        for n in range(len(files)):
            colPrint(
                "{}: {}".format(n, re.sub(r"\.txt$", "", files[n].name)), colors["menu"]
            )
        return selectFile(files[getNumberInput(len(files) - 1)])
    else:
        with p.open("r", encoding="utf-8") as file:
            line1 = file.readline()
            rest = file.read()
        return (line1, rest)


def instructions():
    with open("interface/instructions.txt", "r", encoding="utf-8") as file:
        colPrint(file.read(), colors["instructions"], False)


def getGenerator():
    colPrint(
        "\nInitializing AI Engine! (This might take a few minutes)\n",
        colors["loading-message"],
    )
    return GPT2Generator(
        generate_num=settings["generate-num"],
        temperature=settings["temp"],
        top_k=settings["top-keks"],
        top_p=settings["top-p"],
    )


if not Path("prompts", "Anime").exists():
    try:
        import pastebin
        pastebin.download_clover_prompts()
    except Exception as e:
        logger.info("Failed to scrape pastebin: %e", e)
        colPrint(
            "Failed to scrape pastebin, possible connection issue.\nTry again later. Continuing without downloading prompts...",
            colors["error"],
        )


class AIPlayer:
    def __init__(self, generator):
        self.generator = generator

    def get_actions(self, prompt):
        suggested_actions = [
            self.get_action(prompt)
            for _ in range(config_act["alternatives"])
        ]
        logger.debug("Suggested actions before filter and dedup %s", suggested_actions)
        # remove short ones
        suggested_actions = [
            s
            for s in suggested_actions
            if len(s) > config_act["min-length"]
        ]
        # remove dups
        suggested_actions = list(set(suggested_actions))
        return suggested_actions

    def get_action(self, prompt):
        result_raw = self.generator.generate_raw(
            prompt, generate_num=config_act["generate-number"], temperature=config_act["temperature"]
        )
        result_raw = standardize_punctuation(result_raw)

        # The generations actions carry on into the next prompt, so lets remove the prompt
        results = result_raw.split("\n")
        results = [s.strip() for s in results]
        results = [s for s in results if len(s) > config_act["min-length"]]
        # Sometimes actions are generated with leading > ! . or ?. Likely the model trying to finish the prompt or start an action.
        result = results[0].strip().lstrip(" >!.?")
        result = cut_trailing_quotes(result)
        logger.debug("full suggested action '%s'. Cropped: '%s'", result_raw, result)

        # Often actions are cropped with sentance fragment, lets remove. Or we could just turn up config_act["generate-number"]
        last_punc = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if (last_punc / len(result)) > 0.7:
            result = result[: - i]
        elif last_punc == len(result):
            pass
        else:
            result += '...'
        return result


def main(generator):
    story_manager = UnconstrainedStoryManager(generator)
    ai_player = AIPlayer(generator)
    print("\n")

    with open("interface/mainTitle.txt", "r", encoding="utf-8") as file:
        colPrint(file.read(), colors["title"])

    with open("interface/subTitle.txt", "r", encoding="utf-8") as file:
        cols = get_terminal_size()[0]
        for line in file:
            line = re.sub(r"\n", "", line)
            line = line[:cols]
            colPrint(
                # re.sub(
                #     r"\|[ _]*\|", lambda x: "\x1B[7m" + x.group(0) + "\x1B[27m", line
                # ),
                line,
                colors["subtitle"],
                False,
            )

    while True:
        if story_manager.story != None:
            del story_manager.story

        print("\n\n")

        colPrint(
            "0: Pick Prompt From File (Default if you type nothing)\n1: Write Custom Prompt",
            colors["menu"],
        )

        if getNumberInput(1) == 1:
            with open(
                Path("interface", "prompt-instructions.txt"), "r", encoding="utf-8"
            ) as file:
                colPrint(file.read(), colors["instructions"], False)
            context = colInput("Context>", colors["main-prompt"], colors["user-text"])
            prompt = colInput("Prompt>", colors["main-prompt"], colors["user-text"])
            filename = colInput(
                "Name to save prompt as? (Leave blank for no save): ",
                colors["query"],
                colors["user-text"],
            )
            filename = re.sub(
                "-$", "", re.sub("^-", "", re.sub("[^a-zA-Z0-9_-]+", "-", filename))
            )
            if filename != "":
                with open(
                    Path("prompts", filename + ".txt"), "w", encoding="utf-8"
                ) as f:
                    # this saves unix style line endings which might be an issue
                    # don't know how to do this properly
                    f.write(context + "\n" + prompt + "\n")
        else:
            context, prompt = selectFile()

        instructions()

        colPrint("\nGenerating story...", colors["loading-message"])

        story_manager.start_new_story(prompt, context=context)
        print("\n")
        colPrint(str(story_manager.story), colors["ai-text"])

        while True:
            # Generate suggested actions
            if int(config_act["alternatives"]) > 0:

                action_prompt = (
                    story_manager.story.results[-1]
                    if story_manager.story.results
                    else "\nWhat do you do now?"
                )
                suggested_actions = ai_player.get_actions(action_prompt)
                if len(suggested_actions):
                    suggested_actions_enum = [
                        f"{i}> {a}\n" for i, a in enumerate(suggested_actions)
                    ]
                    suggested_action = "".join(suggested_actions_enum)
                    colPrint(
                        "\nSuggested actions:\n" + suggested_action,
                        colors["selection-value"],
                    )
                    print("\n")

            if settings["console-bell"]:
                print("\x07", end="")
            action = colInput("> ", colors["main-prompt"], colors["user-text"])
            setRegex = re.search("^set ([^ ]+) ([^ ]+)$", action)
            if setRegex:
                if setRegex.group(1) in settings:
                    currentSettingValue = settings[setRegex.group(1)]
                    colPrint(
                        "Current Value of {}: {}     Changing to: {}".format(
                            setRegex.group(1), currentSettingValue, setRegex.group(2)
                        )
                    )
                    settings[setRegex.group(1)] = setRegex.group(2)
                    colPrint("Save config file?", colors["query"])
                    colPrint(
                        "Saving an invalid option will corrupt file!", colors["error"]
                    )
                    if (
                        colInput(
                            "y/n? >",
                            colors["selection-prompt"],
                            colors["selection-value"],
                        )
                        == "y"
                    ):
                        with open("config.ini", "w", encoding="utf-8") as file:
                            config.write(file)

                    # Test this
                    k = setRegex.group(1).replace('-', '_')
                    v = setRegex.group(2)
                    if hasattr(story_manager.generator, k):
                        setattr(story_manager.generator, k, v)

                    # # FIXME this is so slow you might as well restart the program, better to just replace variables
                    # gc.collect()
                    # del story_manager.generator
                    # story_manager.generator = getGenerator()
                else:
                    colPrint("Invalid Setting", colors["error"])
                    instructions()
            elif action == "restart":
                break
            elif action == "quit":
                exit()
            elif action == "help":
                instructions()
            elif action == "print":
                print("\nPRINTING\n")
                colPrint(str(story_manager.story), colors["print-story"])
            elif action == "revert":

                if len(story_manager.story.actions) is 0:
                    colPrint("You can't go back any farther. ", colors["error"])
                    continue

                story_manager.story.actions = story_manager.story.actions[:-1]
                story_manager.story.results = story_manager.story.results[:-1]
                colPrint("Last action reverted. ", colors["message"])
                if len(story_manager.story.results) > 0:
                    colPrint(story_manager.story.results[-1], colors["ai-text"])
                else:
                    colPrint(story_manager.story.story_start, colors["ai-text"])
                continue

            else:
                if action == "":
                    # Use a random suggested action
                    action = random.sample(suggested_actions, 1)[0]
                elif action in [str(i) for i in range(len(suggested_actions))]:
                    action = suggested_actions[int(action)]

                # Roll a 20 sided dice to make things interesting
                d = random.randint(1, 20)
                logger.info("d20 roll %s", d)
                if action[0] == '"':
                    if d == 1:
                        verbs_say_d01 = ["mumble", "prattle", "incoherently say", "whine", "ramble", "wheeze"]
                        verb = random.sample(verbs_say_d01, 1)[0]
                        action = "You "+verb+" " + action
                    elif d == 20:
                        verbs_say_d20 = ["persuasively", "expertly", "conclusively", "dramatically", "adroitly", "aptly"]
                        verb = random.sample(verbs_say_d20, 1)[0]
                        action = "You "+verb+" say " + action
                    else:
                        action = "You say " + action
                else:
                    action = action.strip()
                    action = action[0].lower() + action[1:]
                    if "You" not in action[:6] and "I" not in action[:6]:
                        # roll a d20
                        if d == 1:
                            verb_action_d01 = ["disastrously", "incompetently", "dangerously", "stupidly", "horribly", "miserably", "sadly"]
                            verb = random.sample(verb_action_d01, 1)[0]
                            action = "You "+verb+" fail to " + action
                        elif d < 5:
                            action = "You start to " + action
                        elif d < 10:
                            action = "You attempt to " + action
                        elif d < 15:
                            action = "You try to " + action
                        elif d < 20:
                            action = "You " + action
                        else:
                            action = "You successfully " + action

                    if action[-1] not in [".", "?", "!"]:
                        action = action + "."

                    action = first_to_second_person(action)

                    action = "\n> " + action + "\n"

                result = "\n" + story_manager.act(action)
                if len(story_manager.story.results) >= 2:
                    similarity = get_similarity(
                        story_manager.story.results[-1], story_manager.story.results[-2]
                    )
                    if similarity > 0.9:
                        story_manager.story.actions = story_manager.story.actions[:-2]
                        story_manager.story.results = story_manager.story.results[:-2]
                        colPrint(
                            "Woops that action caused the model to start looping. Try a different action to prevent that.",
                            colors["error"],
                        )
                        continue

                if player_won(result):
                    colPrint(result + "\n CONGRATS YOU WIN", colors["message"])
                    break
                elif player_died(result):
                    colPrint(result, colors["ai-text"])
                    colPrint("YOU DIED. GAME OVER", colors["error"])
                    colPrint(
                        "\nOptions:\n0)Start a new game\n1)\"I'm not dead yet!\" (If you didn't actually die)",
                        colors["menu"],
                    )
                    choice = getNumberInput(1)
                    if choice == 0:
                        break
                    else:
                        colPrint("Sorry about that...where were we?", colors["query"])
                        colPrint(result, colors["ai-text"])

                else:
                    colPrint("> " + action, colors["user-text"])
                    colPrint(result, colors["ai-text"])


if __name__ == "__main__":
    generator = getGenerator()

    main(generator)
