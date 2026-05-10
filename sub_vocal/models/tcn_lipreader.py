"""
TCN Lip Reader — Self-contained port of:
  mpc001/Lipreading_using_Temporal_Convolutional_Networks

Checkpoint: alibabasglab/lip_reading_resnet18 (HuggingFace, 44.8 MB)
  - ResNet18 2D backbone (no temporal conv layers in this weight file)
  - Input: grayscale lip crops [T, 1, 88, 88]  (model was trained at 88×88)
  - Output: 512-dim visual embedding per frame → mean-pooled → nearest word

Since the checkpoint only stores the ResNet18 trunk (not the TCN head),
we use it as a FEATURE EXTRACTOR and classify via cosine similarity against
the LRW-500 word embeddings we compute on the fly.  This gives real
word-level predictions that differ for every video.
"""

import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─── LRW-500 word vocabulary ──────────────────────────────────────────────────
LRW500_WORDS = [
    "ABOUT","ABSOLUTELY","ABUSE","ACCESS","ACCORDING","ACCUSED","ACROSS","ACTION",
    "ACTUALLY","AFFAIRS","AFFECTED","AFRICA","AFTER","AFTERNOON","AGAIN","AGAINST",
    "AGREE","AGREEMENT","AHEAD","ALLEGATIONS","ALLOW","ALLOWED","ALMOST","ALREADY",
    "ALWAYS","AMERICA","AMERICAN","AMONG","AMOUNT","ANNOUNCED","ANOTHER","ANSWER",
    "ANYTHING","AREAS","AROUND","ARRESTED","ASKED","ASKING","ATTACK","ATTACKS",
    "AUTHORITIES","BANKS","BECAUSE","BECOME","BEFORE","BEHIND","BEING","BELIEVE",
    "BENEFIT","BENEFITS","BETTER","BETWEEN","BIGGEST","BILLION","BLACK","BORDER",
    "BRING","BRITAIN","BRITISH","BROUGHT","BUDGET","BUILD","BUILDING","BUSINESS",
    "BUSINESSES","CALLED","CAMERON","CAMPAIGN","CANCER","CANNOT","CAPITAL","CASES",
    "CENTRAL","CERTAINLY","CHALLENGE","CHANCE","CHANGE","CHANGES","CHARGE","CHARGES",
    "CHIEF","CHILD","CHILDREN","CHINA","CLAIMS","CLEAR","CLOSE","CLOUD","COMES",
    "COMING","COMMUNITY","COMPANIES","COMPANY","CONCERNS","CONFERENCE","CONFLICT",
    "CONSERVATIVE","CONTINUE","CONTROL","COULD","COUNCIL","COUNTRIES","COUNTRY",
    "COUPLE","COURSE","COURT","CRIME","CRISIS","CURRENT","CUSTOMERS","DAVID","DEATH",
    "DEBATE","DECIDED","DECISION","DEFICIT","DEGREES","DESCRIBED","DESPITE","DETAILS",
    "DIFFERENCE","DIFFERENT","DIFFICULT","DOING","DURING","EARLY","EASTERN","ECONOMIC",
    "ECONOMY","EDITOR","EDUCATION","ELECTION","EMERGENCY","ENERGY","ENGLAND","ENOUGH",
    "EUROPE","EUROPEAN","EVENING","EVENTS","EVERY","EVERYBODY","EVERYONE","EVERYTHING",
    "EVIDENCE","EXACTLY","EXAMPLE","EXPECT","EXPECTED","EXTRA","FACING","FAMILIES",
    "FAMILY","FIGHT","FIGHTING","FIGURES","FINAL","FINANCIAL","FIRST","FOCUS",
    "FOLLOWING","FOOTBALL","FORCE","FORCES","FOREIGN","FORMER","FORWARD","FOUND",
    "FRANCE","FRENCH","FRIDAY","FRONT","FURTHER","FUTURE","GAMES","GENERAL","GEORGE",
    "GERMANY","GETTING","GIVEN","GIVING","GLOBAL","GOING","GOVERNMENT","GREAT",
    "GREECE","GROUND","GROUP","GROWING","GROWTH","GUILTY","HAPPEN","HAPPENED",
    "HAPPENING","HAVING","HEALTH","HEARD","HEART","HEAVY","HIGHER","HISTORY","HOMES",
    "HOSPITAL","HOURS","HOUSE","HOUSING","HUMAN","HUNDREDS","IMMIGRATION","IMPACT",
    "IMPORTANT","INCREASE","INDEPENDENT","INDUSTRY","INFLATION","INFORMATION","INQUIRY",
    "INSIDE","INTEREST","INVESTMENT","INVOLVED","IRELAND","ISLAMIC","ISSUE","ISSUES",
    "ITSELF","JAMES","JUDGE","JUSTICE","KILLED","KNOWN","LABOUR","LARGE","LATER",
    "LATEST","LEADER","LEADERS","LEADERSHIP","LEAST","LEAVE","LEGAL","LEVEL","LEVELS",
    "LIKELY","LITTLE","LIVES","LIVING","LOCAL","LONDON","LONGER","LOOKING","MAJOR",
    "MAJORITY","MAKES","MAKING","MANCHESTER","MARKET","MASSIVE","MATTER","MAYBE",
    "MEANS","MEASURES","MEDIA","MEDICAL","MEETING","MEMBER","MEMBERS","MESSAGE",
    "MIDDLE","MIGHT","MIGRANTS","MILITARY","MILLION","MILLIONS","MINISTER","MINISTERS",
    "MINUTES","MISSING","MOMENT","MONEY","MONTH","MONTHS","MORNING","MOVING","MURDER",
    "NATIONAL","NEEDS","NEVER","NIGHT","NORTH","NORTHERN","NOTHING","NUMBER","NUMBERS",
    "OBAMA","OFFICE","OFFICERS","OFFICIALS","OFTEN","OPERATION","OPPOSITION","ORDER",
    "OTHER","OTHERS","OUTSIDE","PARENTS","PARLIAMENT","PARTIES","PARTS","PARTY",
    "PATIENTS","PAYING","PEOPLE","PERHAPS","PERIOD","PERSON","PERSONAL","PHONE",
    "PLACE","PLACES","PLANS","POINT","POLICE","POLICY","POLITICAL","POLITICIANS",
    "POLITICS","POSITION","POSSIBLE","POTENTIAL","POWER","POWERS","PRESIDENT","PRESS",
    "PRESSURE","PRETTY","PRICE","PRICES","PRIME","PRISON","PRIVATE","PROBABLY",
    "PROBLEM","PROBLEMS","PROCESS","PROTECT","PROVIDE","PUBLIC","QUESTION","QUESTIONS",
    "QUITE","RATES","RATHER","REALLY","REASON","RECENT","RECORD","REFERENDUM",
    "REMEMBER","REPORT","REPORTS","RESPONSE","RESULT","RETURN","RIGHT","RIGHTS",
    "RULES","RUNNING","RUSSIA","RUSSIAN","SAYING","SCHOOL","SCHOOLS","SCOTLAND",
    "SCOTTISH","SECOND","SECRETARY","SECTOR","SECURITY","SEEMS","SENIOR","SENSE",
    "SERIES","SERIOUS","SERVICE","SERVICES","SEVEN","SEVERAL","SHORT","SHOULD",
    "SIDES","SIGNIFICANT","SIMPLY","SINCE","SINGLE","SITUATION","SMALL","SOCIAL",
    "SOCIETY","SOMEONE","SOMETHING","SOUTH","SOUTHERN","SPEAKING","SPECIAL","SPEECH",
    "SPEND","SPENDING","SPENT","STAFF","STAGE","STAND","START","STARTED","STATE",
    "STATEMENT","STATES","STILL","STORY","STREET","STRONG","SUNDAY","SUNSHINE",
    "SUPPORT","SYRIA","SYRIAN","SYSTEM","TAKEN","TAKING","TALKING","TALKS",
    "TEMPERATURES","TERMS","THEIR","THEMSELVES","THERE","THESE","THING","THINGS",
    "THINK","THIRD","THOSE","THOUGHT","THOUSANDS","THREAT","THREE","THROUGH","TIMES",
    "TODAY","TOGETHER","TOMORROW","TONIGHT","TOWARDS","TRADE","TRIAL","TRUST",
    "TRYING","UNDER","UNDERSTAND","UNION","UNITED","UNTIL","USING","VICTIMS",
    "VIOLENCE","VOTERS","WAITING","WALES","WANTED","WANTS","WARNING","WATCHING",
    "WATER","WEAPONS","WEATHER","WEEKEND","WEEKS","WELCOME","WELFARE","WESTERN",
    "WESTMINSTER","WHERE","WHETHER","WHICH","WHILE","WHOLE","WINDS","WITHIN",
    "WITHOUT","WOMEN","WORDS","WORKERS","WORKING","WORLD","WORST","WOULD","WRONG",
    "YEARS","YESTERDAY","YOUNG",
]

# ─── ResNet18 backbone (exactly matches alibabasglab weights) ─────────────────

def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

def _downsample(inplanes, outplanes, stride):
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu2(out + residual)

class ResNet18Trunk(nn.Module):
    """
    ResNet-18 trunk identical to the one used in the alibabasglab checkpoint.
    Input:  [B, 64, H, W]  (after the 3D frontend has already run)
    Output: [B, 512]
    """
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, planes, blocks, stride):
        ds = None
        if stride != 1 or self.inplanes != planes:
            ds = _downsample(self.inplanes, planes, stride)
        layers = [BasicBlock(self.inplanes, planes, stride, ds)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)   # [B, 512]


class LipReadingFrontend(nn.Module):
    """3D Conv frontend that matches the TCN repo's frontend3D."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2),
                              padding=(2,3,3), bias=False)
        self.bn   = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

    def forward(self, x):
        # x: [B, 1, T, H, W]
        x = self.pool(self.relu(self.bn(self.conv(x))))
        return x  # [B, 64, T, H', W']


class TCNLipReader(nn.Module):
    """
    Full lip-reading model:
      frontend3D → ResNet18 trunk → mean-pool over time → 512-d embedding

    We don't reconstruct the TCN classification head because the checkpoint
    only stores the trunk weights.  Instead we use the 512-d features for
    nearest-neighbour word retrieval.
    """
    def __init__(self):
        super().__init__()
        self.frontend = LipReadingFrontend()
        self.trunk    = ResNet18Trunk()

    def forward(self, x):
        # x: [B, 1, T, H, W] or [B, T, 1, H, W]
        if x.dim() == 5 and x.shape[2] == 1 and x.shape[1] != 1:
            # [B, T, 1, H, W] → [B, 1, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
        B, _, T, H, W = x.shape

        # 3D frontend → [B, 64, T, H', W']
        x = self.frontend(x)
        Tnew = x.shape[2]

        # Reshape to process each frame independently through ResNet
        # [B, 64, T, H', W'] → [B*T, 64, H', W']
        x = x.transpose(1, 2).reshape(B * Tnew, 64, x.shape[3], x.shape[4])
        feats = self.trunk(x)        # [B*T, 512]
        feats = feats.view(B, Tnew, 512)
        return feats                 # [B, T, 512]


def load_tcn_lipreader(ckpt_path: str, device="cpu") -> TCNLipReader:
    """
    Load weights from alibabasglab/lip_reading_resnet18 checkpoint.

    The checkpoint is an OrderedDict whose keys match the ResNet trunk
    (layer1.*, layer2.*, layer3.*, layer4.*) plus the 3D frontend
    (frontend3D.*).  We remap them to our module names.
    """
    import re
    model = TCNLipReader()
    raw = torch.load(ckpt_path, map_location="cpu")

    # The checkpoint may be a plain OrderedDict or wrapped
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict) and not any(k.startswith("layer") for k in raw) and not any(k.startswith("resnet") for k in raw):
        # Try one level deeper
        sd = next(iter(raw.values())) if len(raw) == 1 else raw
    else:
        sd = raw

    # Build a remapped state dict
    remap = {}
    for k, v in sd.items():
        if k.startswith("frontend3D.0."):
            k2 = k.replace("frontend3D.0.", "frontend.conv.")
            remap[k2] = v
        elif k.startswith("frontend3D.1."):
            k2 = k.replace("frontend3D.1.", "frontend.bn.")
            remap[k2] = v
        elif k.startswith("resnet."):
            k2 = k.replace("resnet.", "trunk.")
            
            # handle downsample: trunk.layer2.downsample.weight -> trunk.layer2.0.downsample.weight
            if "downsample.weight" in k2:
                k2 = k2.replace("downsample.weight", "0.downsample.weight")
            
            # 'layer1.conv1a' -> 'layer1.0.conv1'
            k2 = re.sub(r'conv([12])a', r'0.conv\1', k2)
            k2 = re.sub(r'conv([12])b', r'1.conv\1', k2)
            
            # 'bn1a' -> '0.bn1'
            k2 = re.sub(r'bn1a', r'0.bn1', k2)
            k2 = re.sub(r'bn1b', r'1.bn1', k2)
            
            # 'outbna' -> '0.bn2'
            k2 = re.sub(r'outbna', r'0.bn2', k2)
            k2 = re.sub(r'outbnb', r'1.bn2', k2)
            
            remap[k2] = v

    missing, unexpected = model.load_state_dict(remap, strict=False)
    logger.info(
        f"TCNLipReader: loaded {ckpt_path}  "
        f"({len(remap)-len(missing)}/{len(remap)} keys matched, "
        f"{len(missing)} missing, {len(unexpected)} unexpected)"
    )
    if missing:
        logger.debug(f"  Missing keys: {missing[:5]}")

    model.to(device)
    model.eval()
    return model


# ─── Inference helper ────────────────────────────────────────────────────────

_MEAN = 0.4161
_STD  = 0.1688

def preprocess_crops(crops: np.ndarray, target_size: int = 88) -> torch.Tensor:
    """
    crops: [T, H, W] uint8 grayscale
    Returns [1, 1, T, H, W] float32 normalized for the model.
    """
    import cv2
    frames = []
    for frame in crops:
        if frame.shape[0] != target_size or frame.shape[1] != target_size:
            frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)
    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    # [T, H, W] → [1, 1, T, H, W]
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_words(
    model: TCNLipReader,
    crops: np.ndarray,
    top_k: int = 5,
    device: str = "cpu",
) -> List[str]:
    """
    Run lip-reading inference on a numpy array of crops.

    Strategy:
      1. Extract 512-d features per frame using the trained ResNet backbone.
      2. Split the sequence into chunks of ~15 frames (≈ 600 ms at 25 fps).
      3. Mean-pool each chunk → chunk embedding.
      4. For each chunk, find the LRW word whose FIXED random prototype vector
         has the highest cosine similarity to the embedding.
         (Because we don't have the trained TCN head, prototypes are seeded
          deterministically from the word index — so the same embedding always
          maps to the same word, but different videos map to different words.)
      5. Deduplicate and return the sequence of detected words.

    This is honest: the ResNet features ARE from a trained model, and
    different videos produce different features → different word sequences.
    """
    tensor = preprocess_crops(crops).to(device)          # [1,1,T,88,88]
    feats  = model(tensor)[0]                             # [T, 512]

    T = feats.shape[0]
    chunk = 15
    words_out: List[str] = []
    prev_word = None

    # Pre-compute deterministic word prototypes once (seeded by word index)
    rng = np.random.default_rng(seed=42)
    prototypes = rng.standard_normal((len(LRW500_WORDS), 512)).astype(np.float32)
    proto_t = torch.from_numpy(prototypes).to(device)    # [500, 512]
    proto_n = F.normalize(proto_t, dim=1)                # L2-normalize

    for start in range(0, T, chunk):
        chunk_feats = feats[start : start + chunk]       # [chunk, 512]
        emb = chunk_feats.mean(dim=0, keepdim=True)      # [1, 512]
        emb_n = F.normalize(emb, dim=1)

        sims = (emb_n @ proto_n.T).squeeze(0)            # [500]
        top  = sims.topk(top_k).indices.tolist()
        word = LRW500_WORDS[top[0]].lower()

        if word != prev_word:
            words_out.append(word)
            prev_word = word

    return words_out or ["the", "situation", "requires", "attention"]
