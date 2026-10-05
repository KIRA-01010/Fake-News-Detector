"""
generate_dataset.py
--------------------
Builds a demo dataset (news_dataset.csv) of labeled REAL / FAKE news text.

This is a SYNTHETIC dataset meant to let the whole project run end-to-end
out of the box. The patterns are realistic (fake news tends to use
sensational, emotional, clickbait language; real news tends to be neutral
and attributed) but it is not a substitute for a real-world dataset.

For a production-quality model, replace this file's output with a real
dataset such as the Kaggle "Fake and Real News Dataset"
(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- just make sure the final CSV has the same two columns: text, label
(label values: REAL / FAKE).
"""

import random
import csv

random.seed(42)

topics = [
    "the economy", "a new vaccine", "the presidential election", "climate policy",
    "a celebrity scandal", "a tech company", "the stock market", "a natural disaster",
    "a sports championship", "a government scandal", "a scientific discovery",
    "a local school district", "a new smartphone", "a national park", "a bank",
    "the housing market", "a university study", "an airline", "a food recall",
    "a space mission", "a cybersecurity breach", "an energy company", "a court ruling",
    "a music festival", "a public health warning", "an immigration policy",
    "a wildlife conservation project", "a transportation strike", "a historic building",
    "a charity organization",
]

people_titles = [
    "the mayor", "a spokesperson", "the CEO", "a senior official", "researchers",
    "the governor", "a police spokesperson", "the health minister", "scientists",
    "a company representative", "the university dean", "a federal judge",
    "local officials", "the department secretary", "an economist",
]

real_templates = [
    "According to {source}, {topic} saw measurable changes this quarter, {title} confirmed in a statement released on Tuesday.",
    "{title} announced updated figures regarding {topic} during a press briefing, citing data collected over the past six months.",
    "A report published by {source} outlines new findings about {topic}, based on interviews with {title} and independent analysts.",
    "Officials confirmed that {topic} will be reviewed next month after {title} presented preliminary data to the committee.",
    "{source} reports that {title} discussed {topic} at a scheduled hearing, noting that further analysis is still ongoing.",
    "A new study published in a peer-reviewed journal examines {topic}, with {title} noting the results are consistent with prior research.",
    "{title} said in an interview with {source} that {topic} remains under review, and that additional details will be released later this year.",
    "Data released by {source} shows a gradual trend in {topic}, which {title} described as consistent with expectations.",
    "The committee overseeing {topic} met on Thursday, where {title} outlined next steps based on the latest available data.",
    "{source} confirmed that {title} will hold a follow-up briefing on {topic} once the full report is finalized.",
]

fake_templates = [
    "You WON'T BELIEVE what {title} just revealed about {topic} — doctors and experts are FURIOUS!!!",
    "SHOCKING: Secret documents PROVE {topic} was a total scam all along, insiders say the truth is being hidden from you!",
    "BREAKING!!! {title} secretly ADMITS {topic} is a hoax, mainstream media REFUSES to report this!",
    "This ONE simple fact about {topic} that {title} doesn't want you to know will change your life forever!",
    "URGENT WARNING: {topic} is being used to control the public, and {title} is covering it up right now!!!",
    "Experts are STUNNED after {title} exposed the shocking truth behind {topic} that nobody is talking about.",
    "You need to see this before it gets DELETED — {title} leaks explosive secret about {topic}!!!",
    "The mainstream media doesn't want you to know this ONE crazy fact about {topic}, according to anonymous insiders.",
    "MIRACLE breakthrough about {topic} that {title} tried to bury is finally going viral — share before it's banned!",
    "Insiders CONFIRM {topic} is part of a massive cover-up, and {title} has been lying to the public for years!!!",
]

sources = [
    "Reuters", "the Associated Press", "a local news outlet", "the regional bureau",
    "an official press release", "the department's newsletter", "a national broadcaster",
    "the city council's report",
]


def make_row(is_fake: bool):
    topic = random.choice(topics)
    title = random.choice(people_titles)
    source = random.choice(sources)
    template = random.choice(fake_templates if is_fake else real_templates)
    text = template.format(topic=topic, title=title, source=source)
    return text, "FAKE" if is_fake else "REAL"


def main(n_per_class: int = 400, out_path: str = "data/news_dataset.csv"):
    rows = []
    seen = set()
    while len(rows) < n_per_class * 2:
        is_fake = len(rows) % 2 == 0
        text, label = make_row(is_fake)
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, label))

    random.shuffle(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
