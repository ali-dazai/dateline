"""Build data/testset.csv from period answers grounded in quoted 1912 articles.

Every row cites articles with a verbatim (OCR) quote. The script refuses to write a row unless
each quote is found in its article (rapidfuzz partial_ratio >= 90) and the article is dated on
or before the row's cutoff. Input: data/questions.csv (questions, cutoffs, future facts).
The period answers were drafted from the article text with an AI assistant (Claude, 2026-11-02)
and are accepted only through this quote check; verified_by records that, and `notes` keeps
the quotes so anyone can re-check a row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "questions.csv"
OUT = ROOT / "data" / "testset.csv"
VERIFIED_BY = "claude-article-check (2026-11-02)"

A = {  # short names for the article ids used below
    "madison13": "1912-04-13_p1_sn99062034_00279523088_1912041301_0631_13",
    "irish13": "1912-04-13_p3_sn90059959_00271760061_1912041301_0102_9",
    "trib11": "1912-04-11_p1_sn83030214_00206531526_1912041101_0187_2",
    "argus15": "1912-04-15_p1_sn92053934_00202198351_1912041501_0673_18",
    "santafe15": "1912-04-15_p1_sn84020630_00415620081_1912041501_0716_21",
    "imperial15": "1912-04-15_p1_sn92070146_00414189404_1912041501_0767_2",
    "gatecity15": "1912-04-15_p1_sn83025182_00295877339_1912041501_0539_4",
    "marshall15": "1912-04-15_p1_sn85049554_00295875744_1912041501_0678_33",
    "bridgeport15": "1912-04-15_p1_sn84022472_00295866792_1912041501_0164_21",
    "chickasha15": "1912-04-15_p1_sn86090528_00295864266_1912041501_0998_46",
    "barre16": "1912-04-16_p1_sn91066782_00415628717_1912041601_0112_27",
    "ocala16": "1912-04-16_p2_sn84027621_00295865854_1912041601_0420_12",
    "omaha16": "1912-04-16_p1_sn99021999_00212470612_1912041601_0448_10",
    "fairmont16": "1912-04-16_p1_sn86092557_00415660261_1912041601_0710_3",
    "evestar16": "1912-04-16_p2_sn83045462_00280658030_1912041601_0112_78",
    "kennebec16": "1912-04-16_p12_sn82014248_00513682700_1912041601_1143_26",
    "evestar15": "1912-04-15_p1_sn83045462_00280658030_1912041501_0085_13",
    "gforks15": "1912-04-15_p1_sn85042373_00295863122_1912041501_0747_7",
    "seattle16": "1912-04-16_p6_sn87093407_00200290586_1912041601_0120_9",
    "washherald16": "1912-04-16_p2_sn83045433_00294558081_1912041601_0114_10",
    "evestar16b": "1912-04-16_p25_sn83045462_00280658030_1912041601_0135_15",
    "washtimes15": "1912-04-15_p1_sn84026749_00294557696_1912041501_0150_9",
    "perth15": "1912-04-15_p1_sn85035720_00383340007_1912041502_0151_8",
    "bham16": "1912-04-16_p1_sn85038485_00340584053_1912041601_0888_21",
    "palladium16": "1912-04-16_p2_sn86058226_00271744948_1912041601_0125_16",
    "salem16": "1912-04-16_p1_sn99063957_00200298615_1912041601_0584_56",
    "norwich16": "1912-04-16_p1_sn82014086_00295866585_1912041601_0421_14",
    "roundup19": "1912-04-19_p1_sn86075094_00295860935_1912041901_0134_25",
    "bryan19": "1912-04-19_p1_sn86088651_00202191150_1912041901_0500_5",
    "palladium15": "1912-04-15_p1_sn86058226_00271744948_1912041501_0116_10",
    "detroit17": "1912-04-17_p1_sn83016689_00279551886_1912041701_0779_8",
    "evestar18": "1912-04-18_p2_sn83045462_00280658030_1912041801_0159_16",
    "palladium16b": "1912-04-16_p2_sn86058226_00271744948_1912041601_0125_6",
    "lakeland16": "1912-04-16_p1_sn95047222_00415663304_1912041601_1184_34",
    "atlanta19": "1912-04-19_p1_sn86090947_00393344155_1912041901_0285_2",
    "pensacola19": "1912-04-19_p2_sn87062268_00415662889_1912041901_0492_10",
    "sentinel10": "1912-04-10_p1_sn89051285_00414212748_1912041001_0375_9",
    "norwich11": "1912-04-11_p4_sn82014086_00295866585_1912041101_0384_90",
    "calumet15": "1912-04-15_p4_sn86086633_00296024569_1912041501_0139_8",
    "alaska15": "1912-04-15_p1_sn96060002_00279526338_1912041501_0127_40",
    "detroit10": "1912-04-10_p1_sn83016689_00279551886_1912041001_0639_5",
    "tacoma15": "1912-04-15_p1_sn88085187_00211108320_1912041501_0954_12",
    "hope18": "1912-04-18_p2_sn87096037_00295864011_1912041801_0657_7",
    "eureka13": "1912-04-13_p1_sn86076201_0027955405A_1912041301_0113_12",
    "bennington17": "1912-04-17_p4_sn95066012_00202199525_1912041701_0073_13",
    "evestar16q": "1912-04-16_p2_sn83045462_00280658030_1912041601_0112_13",
    "bismarck15": "1912-04-15_p8_sn85042242_00212478490_1912041501_0751_21",
    "nyt12": "1912-04-12_p6_sn85038615_00175031911_1912041201_0954_11",
    "trib18": "1912-04-18_p8_sn83030214_00206531526_1912041801_0340_16",
    "eveworld19": "1912-04-19_p1_sn83030193_00211100163_1912041901_0796_17",
    "saltlake14": "1912-04-14_p1_sn83045396_print_1912041401_0287_3",
    "tonopah16": "1912-04-16_p1_sn86076142_00415626988_1912041601_0359_19",
    "richmond14": "1912-04-14_p29_sn85038615_00175031911_1912041401_0999_12",
    "trib19": "1912-04-19_p16_sn83030214_00206531526_1912041901_0362_11",
    "bham19": "1912-04-19_p2_sn85038485_00340584053_1912041901_0935_26",
    "omaha14": "1912-04-14_p1_sn99021999_00212470612_1912041401_0392_19",
    "cairo14": "1912-04-14_p1_sn93055779_00295872664_1912041401_0749_7",
    "brownsville15": "1912-04-15_p3_sn86063730_00414214010_1912041501_0206_4",
    "trib12": "1912-04-12_p11_sn83030214_00206531526_1912041201_0211_37",
    "sanjuan19": "1912-04-19_p3_sn88085190_00211107996_1912041901_0021_16",
    "trib19b": "1912-04-19_p16_sn83030214_00206531526_1912041901_0362_8",
    "eveworld16": "1912-04-16_p1_sn83030193_00211100163_1912041601_0719_10",
    "newark16": "1912-04-16_p1_sn91064011_00332896192_1912041601_0639_3",
    "sentinel16": "1912-04-16_p1_sn86066870_00513680326_1912041601_0187_3",
}

# (question_id, cutoff) -> (period_answer, [(article key, verbatim quote), ...])
ROWS: dict[tuple[str, str], tuple[str, list[tuple[str, str]]]] = {
    ("Q01", "1912-04-14"): (
        "Nothing has gone wrong as far as the papers know: the new White Star liner Titanic, the largest ship afloat, "
        "sailed from Southampton on April 10 on her maiden voyage to New York (narrowly missing the liner New York as she left port).",
        [("irish13", "sailed from Southampton today on her maid"),
         ("madison13", "started from this port on her maiden voyage to New York"),
         ("trib11", "the terrific suction of the Titanic's screws dragged the New York from her moorings")]),
    ("Q01", "1912-04-15"): (
        "She struck an iceberg off the Newfoundland banks on Sunday night. Reports conflict: many papers say her passengers were "
        "transferred safely to the Carpathia and Parisian and that the damaged ship was afloat, heading or being towed to Halifax; "
        "others report her sinking.",
        [("argus15", "II is probable all pas sensers on the Titanic are safe"),
         ("imperial15", "is being towed to port by the liner Virginian"),
         ("santafe15", "was in a sinking condition")]),
    ("Q01", "1912-04-16"): (
        "The Titanic sank early Monday, about 2:20 a.m. The Carpathia found only boats and wreckage and picked up about 675 survivors "
        "(later reports say more), nearly all women and children; well over a thousand people are feared lost. The Carpathia is "
        "bringing the survivors to New York.",
        [("barre16", "Titanic sank about 2:20 a. m."),
         ("barre16", "Found boats and wreckage only"),
         ("ocala16", "only 675 are known to have been saved")]),
    ("Q02", "1912-04-15"): (
        "Most reports say yes or probably: passengers were being transferred safely to the Carpathia and Parisian, and one dispatch "
        "says this was done without loss of life. Some papers still report the ship sinking, so the picture is unconfirmed.",
        [("argus15", "II is probable all pas sensers on the Titanic are safe"),
         ("imperial15", "accomplished without loss of life or injury"),
         ("marshall15", "the transferring of passengers is being carried on successfully and safely")]),
    ("Q02", "1912-04-16"): (
        "No. The Titanic sank; the Carpathia has only about 675 to 868 survivors, mostly women and children, and well over a "
        "thousand are feared lost. The White Star Line conceded a horrible loss of life; some hope remains that the Virginian or "
        "Parisian picked up others.",
        [("ocala16", "only 675 are known to have been saved"),
         ("omaha16", "conceded that there had been \"a horrible loss of life\""),
         ("kennebec16", "First reports were that the Carpathia had saved but 675 persons")]),
    ("Q03", "1912-04-15"): (
        "Reports name the Cunard liner Carpathia and the Allan liner Parisian as having taken passengers off, with the Virginian "
        "standing by to tow the Titanic to Halifax and the Baltic approaching.",
        [("bridgeport15", "PASSENGERS OF TITANIC TRANSFERRED IN SAFETY TO PARISIAN AND CARPATHIA"),
         ("chickasha15", "the Virginia passed line to the Tianic and eili tow her to Halifax"),
         ("gatecity15", "having transferred Ter passengers to the Partisan and Carpathia")]),
    ("Q03", "1912-04-16"): (
        "The Carpathia, which is carrying the survivors to New York and is expected about Thursday night, April 18. No survivors "
        "have been reported on the Virginian or Parisian.",
        [("omaha16", "was returning to New York with the survivors"),
         ("evestar16", "is due at Sandy Hook about IL o'clock Thursday night"),
         ("kennebec16", "She did not report the presence of any survivors on board")]),
    ("Q04", "1912-04-15"): (
        "No specific word. Colonel and Mrs. John Jacob Astor are listed among the prominent passengers, and the general reports "
        "say the passengers were being transferred safely to other ships.",
        [("evestar15", "COL and Mrs John Jacob Astor"),
         ("gforks15", "PASSENGERS ARE BELIEVED SAFE"),
         ("marshall15", "Col. and Mrs. John Jacob Astor")]),
    ("Q04", "1912-04-16"): (
        "Unclear and conflicting. Mrs. Astor is reported rescued and aboard the Carpathia. Several papers report Colonel Astor "
        "drowned or probably went down with the ship, while at least one headline lists him among those picked up.",
        [("seattle16", "MRS. ASTOR SAVED; COLONEL DROWNED"),
         ("washherald16", "his wife was safe on the Carpathia"),
         ("evestar16b", "John Jacob Astor Among Those Picked Up")]),
    ("Q05", "1912-04-15"): (
        "White Star officials said the ship was unsinkable and in no danger; vice-president P. A. S. Franklin said he had no doubt "
        "the Titanic would be safely towed to port, and that the loss of wireless contact was no sign of danger.",
        [("washtimes15", "the great liner was unsinkable"),
         ("perth15", "no doubt but that the Titanic would be safely towed to port"),
         ("washtimes15", "the Interruption OF wireless communication was not significant Of danger")]),
    ("Q05", "1912-04-16"): (
        "Vice-president Franklin conceded a horrible loss of life and said only about 675 had been reported rescued; he said the "
        "money could be replaced but not the lives, and angrily denied that the company had withheld news of the sinking.",
        [("omaha16", "conceded that there had been \"a horrible loss of life\""),
         ("bham16", "We replace the money he added but not the lives"),
         ("palladium16", "That is an absolute lie, and those who made the statement know they lied")]),
    ("Q06", "1912-04-16"): (
        "Not known yet. Estimates say well over a thousand are feared lost (figures of roughly 1,200 to 1,800 appear; the Olympic's "
        "message said the loss was likely very heavy), with about 675 to 868 survivors on the Carpathia.",
        [("norwich16", "it is feared more than 1900 persons were lost"),
         ("salem16", "persons u were lost In the wreck OF the liner Titanlc"),
         ("ocala16", "only 675 are known to have been saved")]),
    ("Q06", "1912-04-19"): (
        "After the Carpathia reached New York, official figures put the dead at roughly 1,500 to 1,600 or more (the OCR of the "
        "counts is garbled), including several rescued people who died afterwards; all the missing notable men are now known to be dead.",
        [("roundup19", "the subsequent death of six persons who had been rescued"),
         ("bryan19", "All missing notable men are now known to be dead"),
         ("roundup19", "These Off cial figures greatly magnify the estimates which had come by wirel")]),
    ("Q07", "1912-04-15"): (
        "No report of his death. Papers say Captain Smith and his crew stayed with the ship and hoped to bring her into Halifax; one "
        "dispatch says he and the crew refused to leave their posts.",
        [("palladium15", "Capt. Smith and his crew had hopes of being able to bring her into Halifax"),
         ("imperial15", "Captain Smith and crew of the II tanic refused to leave their posts")]),
    ("Q07", "1912-04-18"): (
        "No. Captain Smith is given up as lost and probably went down with his ship; his widow has posted a message of sympathy "
        "outside the White Star offices.",
        [("evestar16b", "probably went to his grave with his vessel"),
         ("detroit17", "CAPTAIN Smith, GiUen Up As Lost In TITANIC Horror"),
         ("evestar18", "The widow OF Capt. Smith. the commander Of the Titanic, has written pathetic message")]),
    ("Q08", "1912-04-16"): (
        "Probably. The House adopted a resolution of sympathy, and Chairman Alexander of the House merchant marine committee said "
        "Congress would surely look into the Titanic's life-saving equipment and legislate; no formal inquiry has been set up yet.",
        [("palladium16b", "CONGRESS PLANS PROBE OF TRAGEDY"),
         ("palladium16b", "con. gress would surely take cognizance of the life saving facilities of the Titanic"),
         ("lakeland16", "unanimous ly adopted resolution extending sympathy")]),
    ("Q08", "1912-04-19"): (
        "Yes. The Senate passed a resolution for an inquiry; a Commerce Committee subcommittee of seven headed by Senator William "
        "Alden Smith of Michigan went to New York, met White Star officials aboard the Carpathia, and planned to begin hearings with "
        "J. Bruce Ismay among the first witnesses.",
        [("atlanta19", "subcommittee of seven, mead ed by Senator William Alden Smith"),
         ("pensacola19", "would begin the invest CATION OF the loss of the Titanic tomorrow"),
         ("atlanta19", "j. Bruce Ismay will be among the first asked to testify")]),
    ("Q09", "1912-04-12"): (
        "Roosevelt has momentum: he beat President Taft about two and a half to one in the Illinois presidential primary (April 9) "
        "and took most of its delegates, but Taft still leads in delegates overall and the contest continues.",
        [("sentinel10", "Theodore Roosevelt led President Taft about two and half to one"),
         ("norwich11", "The result of the illinois primaries provided two surprises")]),
    ("Q09", "1912-04-15"): (
        "Roosevelt won again in the Pennsylvania primary (April 13), taking most of the state's 76 delegates, after Illinois. "
        "Taft still had far more delegates before these primaries, so the race remains open.",
        [("calumet15", "Apparently Colonel Roosevelt Ia. triumphed in Pennsylvania"),
         ("calumet15", "Mr. Taft had three times as many dele gates as the ex-rough rider"),
         ("alaska15", "ROOSEVELT WINS OVER TAFT IN PENNSYLVANIA")]),
    ("Q10", "1912-04-15"): (
        "Undecided. Champ Clark beat Woodrow Wilson about two to one in the Illinois primary and is sure of Illinois' delegates, "
        "while Wilson is expected to get Pennsylvania's delegation.",
        [("sentinel10", "Champ Clark's vote in the same primary was approximately twice that of Governor Wilson"),
         ("tacoma15", "Wood row Wilson will ge the solid Pennsylvania delegation")]),
    ("Q10", "1912-04-19"): (
        "Still undecided. Clark leads in instructed delegates, but Wilson has gained, leaving Clark a lead of only about sixty; "
        "many delegates are uninstructed.",
        [("hope18", "Woodrow Wilson Climbs to"),
         ("hope18", "ing Champ Clark a Lead Of Only sixty"),
         ("hope18", "are uninstructed")]),
    ("Q11", "1912-04-15"): (
        "Not alone. One report says Miss Mary Davis flew from London to Paris in an aeroplane (with a pilot) on April 2 and calls her "
        "the first woman to make the trip across the Channel; no woman is reported to have piloted a crossing herself.",
        [("eureka13", "Miss Davis is the first woman aviator to make the trip across the English channel"),
         ("eureka13", "few from London to Paris in an aeroplane on April 2")]),
    ("Q11", "1912-04-18"): (
        "Yes. Miss Harriet Quimby, an American, flew alone from Dover across the Channel on April 16, landing at Hardelot near "
        "Boulogne; she is the first woman to fly the Channel alone.",
        [("evestar16q", "crossed the English channel from Dover this morning"),
         ("bennington17", "Miss Quimby is the first woman to accomplish the feat of fly ing across the channel alone")]),
    ("Q12", "1912-04-17"): (
        "Not yet reported. The Red Sox opened their season on the road in New York on April 11, and Fenway Park is mentioned as "
        "their new grounds, but no home opening has been reported.",
        [("nyt12", "Boston Red Sox won the opening game OF the American League season in New York"),
         ("bismarck15", "It is good for perpetual admission to Fenway park")]),
    ("Q12", "1912-04-20"): (
        "Not yet, as far as these papers report: the Yankees left for Boston to open the Red Sox's new home, but the game in "
        "Boston on April 19 was postponed because of rain.",
        [("trib18", "Team Starts for Boston to Open New Home of Red SoX"),
         ("eveworld19", "game with the Red sox to-day was postponed because of rain")]),
    ("Q13", "1912-04-16"): (
        "The Italo-Turkish war drags on in Tripoli (Italian dirigibles scouting Turkish and Arab positions). The European powers are "
        "preparing mediation, and there is speculation that Italy may try to force the Dardanelles.",
        [("richmond14", "two dirigible bal looms yesterday morning made a nigh"),
         ("tonopah16", "mediation by the powers with the object of settling the TurkoItalian war"),
         ("saltlake14", "Forcing of DardanelleS Would Bring About Tremendous Crash")]),
    ("Q13", "1912-04-19"): (
        "The Italian fleet bombarded the Turkish forts at the entrance to the Dardanelles on April 18 for about two and a half hours, "
        "then withdrew; the forts were only slightly damaged, an Italian warship was reported hit, and Turkey proclaimed a blockade "
        "of the Dardanelles.",
        [("trib19", "twenty-seven Italian warships to-day bombarded the forts"),
         ("bham19", "The porte has proclaimed blockade of the Dardanelles"),
         ("trib19", "the forts were only slightly damaged")]),
    ("Q14", "1912-04-15"): (
        "Mixed. Orozco's rebels hold Chihuahua and won at Jimenez, but they are reported short of ammunition, and a federal force of "
        "several thousand under General Huerta is gathering near Torreon/Mapimi for an advance that Orozco expects within days.",
        [("omaha14", "won the day for the rebel forces under General orozco"),
         ("cairo14", "an advance of federal force, which for three weeks has been mobilizing"),
         ("trib12", "is losing faith in the success of the revolution through being unable 10 obtain supplies of ammunition")]),
    ("Q14", "1912-04-20"): (
        "Unclear. Madero's agent says the situation is improving, with General Huerta going north with a large column and Orozco's "
        "forces short of ammunition; meanwhile rebels are reported to have captured Culiacan, capital of Sinaloa, and Sierra Mojada.",
        [("sanjuan19", "General HUI erta has gone north with large column to give battle to the rebels"),
         ("sanjuan19", "Forces OF Orozco lacking in ammunition"),
         ("trib19b", "the capture yesterday of Cuiacan. the capital of Sinaloa, by rebels")]),
    ("Q15", "1912-04-16"): (
        "Probably not. Isidor Straus and his wife are not on the survivor lists received so far and are reported among the lost, "
        "though an early report had placed him among the survivors.",
        [("eveworld16", "lsidor Straus, Major Archibald Butt, William T. Stead"),
         ("newark16", "Their names were not in the survivors list"),
         ("sentinel16", "the Titanic at first reported Idor straus")]),
}


def main() -> None:
    questions = pd.read_csv(QUESTIONS, dtype=str).fillna("")
    arts = pd.read_parquet(ROOT / "data" / "articles.parquet",
                           columns=["article_id", "lccn", "date", "edition", "headline", "text"])
    arts = arts[arts["article_id"].isin(set(A.values()))].set_index("article_id")
    missing = set(A.values()) - set(arts.index)
    if missing:
        sys.exit(f"articles not found: {sorted(missing)}")

    # Keep the memorised column filled by eval.memorisation if the file is rebuilt.
    memorised = {}
    if OUT.exists():
        prev = pd.read_csv(OUT, dtype=str).fillna("")
        memorised = {(r.question_id, r.cutoff): r.memorised for r in prev.itertuples()}

    problems, out = [], []
    for _, row in questions.iterrows():
        row = row.copy()
        row["memorised"] = memorised.get((row["question_id"], row["cutoff"]), row["memorised"])
        key = (row["question_id"], row["cutoff"])
        if key not in ROWS:
            problems.append(f"{key}: no period answer written")
            continue
        answer, evidence = ROWS[key]
        ids, urls, notes = [], [], []
        for short, quote in evidence:
            aid = A[short]
            art = arts.loc[aid]
            blob = f"{art['headline']} {art['text']}"
            score = fuzz.partial_ratio(quote.lower(), blob.lower())
            day = str(art["date"])[:10]
            if score < 90:
                problems.append(f"{key}: quote not found in {aid} (score {score:.0f}): {quote!r}")
            if day > row["cutoff"]:
                problems.append(f"{key}: {aid} is dated {day}, after the cutoff")
            if aid not in ids:
                ids.append(aid)
                ed = int(str(art["edition"] or 1)) if str(art["edition"]).isdigit() else 1
                urls.append(f"https://www.loc.gov/item/{art['lccn']}/{day}/ed-{ed}/")
            notes.append(f'"{quote}" [{aid}]')
        out.append({**row.to_dict(), "period_answer": answer, "source_article_ids": ";".join(ids),
                    "source_urls": ";".join(urls), "verified_by": VERIFIED_BY,
                    "notes": "Evidence (OCR quotes): " + " | ".join(notes)})
    if problems:
        sys.exit("NOT WRITTEN:\n" + "\n".join(problems))
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(f"wrote {len(out)} rows to {OUT}")


if __name__ == "__main__":
    main()
