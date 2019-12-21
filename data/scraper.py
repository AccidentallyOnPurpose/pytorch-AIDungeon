import json
import time
from pathlib import Path
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


logger = logging.getLogger(__name__)

"""
format of tree is
dict {
    tree_id: tree_id_text
    context: context text?
    first_story_block
    action_results: [act_res1, act_res2, act_res3...]
}

where each action_result's format is:
dict{
    action: action_text
    result: result_text
    action_results: [act_res1, act_res2, act_res3...]
}
"""


class Scraper:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("--binary=/path/to/other/chrome/binary")
        chrome_options.add_argument("--incognito")
        chrome_options.add_argument("--window-size=1920x1080")
        exec_path = "/usr/bin/chromedriver"
        self.driver = webdriver.Chrome(
            chrome_options=chrome_options, executable_path=exec_path
        )
        self.max_depth = 10
        self.end_actions = {
            "End Game and Leave Comments",
            "Click here to End the Game and Leave Comments",
            "See How Well You Did (you can still back-page afterwards if you like)",
            "You have died.",
            "You have died",
            "Epilogue",
            "Save Game",
            "Your quest might have been more successful...",
            "5 - not the best, certainly not the worst",
            "The End! (leave comments on game)",
            "6 - it's worth every cent",
            "You do not survive the journey to California",
            "Quit the game.",
            "7 - even better than Reeses' Cups®",
            "8 - it will bring you enlightenment",
            "End of game! Leave a comment!",
            "Better luck next time",
            "click here to continue",
            "Rating And Leaving Comments",
            "You do not survive your journey to California",
            "Your Outlaw Career has come to an end",
            "Thank you for taking the time to read my story",
            "You have no further part in the story, End Game and Leave Comments",
            "",
            "You play no further part in this story. End Game and Leave Comments",
            "drivers",
            "Alas, poor Yorick, they slew you well",
            "My heart bleeds for you",
            "To End the Game and Leave Comments click here",
            "Call it a day",
            "Check the voicemail.",
            "reset",
            "There's nothing you can do anymore...it's over.",
            "To Be Continued...",
            "Thanks again for taking the time to read this",
            "If you just want to escape this endless story you can do that by clicking here",
            "Boo Hoo Hoo",
            "End.",
            "Pick up some money real quick",
            "",
            "Well you did live a decent amount of time in the Army",
            "End Game",
            "You have survived the Donner Party's journey to California!",
        }
        self.texts = set()

    def GoToURL(self, url):
        self.texts = set()
        self.driver.get(url)
        time.sleep(0.5)

    def GetText(self):
        div_elements = self.driver.find_elements_by_css_selector("div")
        text = div_elements[3].text
        return text

    def GetLinks(self):
        return self.driver.find_elements_by_css_selector("a")

    def GoBack(self):
        self.GetLinks()[0].click()
        time.sleep(0.2)

    def ClickAction(self, links, action_num):
        links[action_num + 4].click()
        time.sleep(0.2)

    def GetActions(self):
        return [link.text for link in self.GetLinks()[4:]]

    def NumActions(self):
        return len(self.GetLinks()) - 4

    def BuildTreeHelper(self, parent_story, action_num, depth, old_actions):
        depth += 1
        action_result = {}

        action = old_actions[action_num]
        print("Action is ", repr(action))
        action_result["action"] = action

        links = self.GetLinks()
        if action_num + 4 >= len(links):
            return None

        self.ClickAction(links, action_num)
        result = self.GetText()
        if result == parent_story or result in self.texts:
            self.GoBack()
            return None

        self.texts.add(result)
        print(len(self.texts))

        action_result["result"] = result

        actions = self.GetActions()
        action_result["action_results"] = []

        for i, action in enumerate(actions):
            if actions[i] not in self.end_actions:
                sub_action_result = self.BuildTreeHelper(result, i, depth, actions)
                if action_result is not None:
                    action_result["action_results"].append(sub_action_result)

        self.GoBack()
        return action_result

    def BuildStoryTree(self, url):
        scraper.GoToURL(url)
        text = scraper.GetText()
        actions = self.GetActions()
        story_dict = {}
        story_dict["tree_id"] = url
        story_dict["context"] = ""
        story_dict["first_story_block"] = text
        story_dict["action_results"] = []

        for i, action in enumerate(actions):
            if action not in self.end_actions:
                action_result = self.BuildTreeHelper(text, i, 0, actions)
                if action_result is not None:
                    story_dict["action_results"].append(action_result)
            else:
                print("done")

        return story_dict


def save_tree(tree, filename):
    with open(filename, "w") as fp:
        json.dump(tree, fp)


scraper = Scraper()

# go to https://www.google.com/search?lr=lang_en&tbs=lr%3Alang_1en&ei=W5H9XaXUOISf4-EPgrSI0AI&q=http%3A%2F%2Fchooseyourstory.com%2Fstory%2Fviewer%2Fdefault.aspx%3FStoryId%3D&oq=http%3A%2F%2Fchooseyourstory.com%2Fstory%2Fviewer%2Fdefault.aspx%3FStoryId%3D&gs_l=psy-ab.3..0i71l6.1400.1400..1600...0.3..0.0.0.......0....1..gws-wiz.R6C0aWPPhMk&ved=0ahUKEwjl7YOa5sXmAhWEzzgGHQIaAioQ4dUDCAo&uact=5
# in js console
# copy(Array.from(document.getElementsByClassName('iUh30')).map(bb=>bb.textContent.split('=').slice(-1)[0]))
# then paste into doc and go to next page
story_ids = [8,
 22,
 253,
 470,
 1046,
 1153,
 1495,
 2280,
 2823,
 4720,
 5466,
 5861,
 6376,
 6823,
 7094,
 7393,
 7397,
 7480,
 7567,
 7770,
 8035,
 8038,
 8040,
 8041,
 8098,
 8557,
 9170,
 9361,
 9411,
 9907,
 9935,
 10183,
 10359,
 10524,
 10634,
 10638,
 10872,
 10885,
 11144,
 11246,
 11274,
 11545,
 11906,
 12992,
 13349,
 13560,
 13875,
 13907,
 14899,
 14976,
 15424,
 15703,
 16489,
 17306,
 17571,
 17801,
 18988,
 19849,
 21858,
 21879,
 23928,
 24743,
 24889,
 25454,
 26558,
 26752,
 27234,
 27392,
 27469,
 28030,
 28838,
 30001,
 31013,
 31014,
 31242,
 31353,
 32415,
 33006,
 34072,
 34688,
 34838,
 34849,
 36594,
 36617,
 36791,
 37696,
 38025,
 38542,
 40954,
 41698,
 42182,
 42204,
 42978,
 43573,
 43744,
 43910,
 43993,
#  44305,
 44543,
 45225,
 45266,
 45375,
 45866,
 48393,
 49180,
 49642,
 50199,
 51926,
 51934,
 51959,
 52887,
 52961,
 53134,
 53186,
 53356,
 53837,
 54011,
 54639,
 54643,
 55043,
 56501,
 56515,
 56742,
 56753,
 57114,
 60128,
 60232,
 60747,
 60772]

base_dir = Path(f"stories/v43/")
base_dir.mkdir(exist_ok=True)

for story_id in story_ids:
    print("****** Extracting Adventure ", story_id, " ***********")
    url = f"http://chooseyourstory.com/story/viewer/default.aspx?StoryId={story_id}"
    story_file = base_dir.joinpath(f"chooseyourstory_{story_id}.json")
    if not story_file.exists():
        try:
            tree = scraper.BuildStoryTree(url)
            save_tree(tree, str(story_file))
        except KeyboardInterrupt as e:
            raise e
        except:
            story_file.open('w').write('{}')
            logger.exception("Failed to scrape story")

print("done")
