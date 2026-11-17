"""Download the InjecAgent test cases (needs network access to github.com)."""
import pathlib, urllib.request
BASE = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/"
FILES = ["test_cases_dh_base.json", "test_cases_ds_base.json", "test_cases_dh_enhanced.json", "test_cases_ds_enhanced.json"]
out = pathlib.Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)
for f in FILES:
    urllib.request.urlretrieve(BASE + f, out / f)
    print("saved", out / f)
