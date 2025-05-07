import random
import torch
import yaml
import pathlib
import numpy as np
import pandas as pd
from string import ascii_uppercase as alphabet

import torch.utils.data as data
from torch.utils.data import Dataset

from PIL import Image

from typing import List, Tuple, Dict, Callable
from easydict import EasyDict
from .patching import Patcher


def _categorize(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    out = df.copy(True)
    for key in keys:
        out[key] = out[key].astype("category").cat.codes
    return out


def _loadPatientsCSV(csvFile: str) -> dict:
    """Load patients data from preprocessed CSV

    Args:
        csvFile (str): Path to preprocessed CSV

    Returns:
        dict: Each key is of the form: SRX where X is the patient letter.
              The dictionnary contains OS [days],PFS [days],Age [years],Sex [0,1],Tumor localization[0,1,2]
    """
    patientsDF = pd.read_csv(
        csvFile,
        sep=";",
        usecols=[
            "Biopsy number",
            "Sex",
            "Age",
            "Tumor localization",
            "PFS",
            "OS",
        ],
        index_col="Biopsy number",
    )

    df = _categorize(patientsDF, ["Sex", "Tumor localization"])

    return df.T.to_dict()


def _loadLineageCSV(csvFile: str) -> dict:
    """Load lineage data for each tumor cell lineage

    Args:
        csvFile (str): Path to lineage preprocessed CSV

    Returns:
        dict: A dictionnary with keys in the format SRXN, X: patient letter, N: lineage.
              The dictionnary contains:
              - radioresistance (float)
              - metabolic profile (0,1,2,3)
              - Neurosphere formation time (days,1000 if never)
              - 2nd generation formation (bool)
              - Neuroshpere formation ratio (float)
    """
    lineageDF = pd.read_csv(
        csvFile,
        usecols=[
            "lineage",
            "spectro",
            "NS_gen_1",
            "NS_gen_2",
            "radioresistance",
            "NS_form_ratio",
        ],
    )
    lineageDF.fillna(-1, inplace=True)
    lineageDF.set_index("lineage", inplace=True)
    fullData = lineageDF.T.to_dict()
    return fullData


class PatchDataset(Dataset):
    TRAIN_VAL_RATIO = 0.9

    def __init__(
        self,
        patchType: str,
        patchRootDir: str,
        patchParams: EasyDict = None,
        train: bool = True,
        transform: Callable = None,
        patientCSV: str = None,
        lineageCSV: str = None,
    ) -> None:
        super().__init__()
        self.patchType = patchType
        self.rootDir = patchRootDir
        self.patchParams = patchParams
        self.train = train
        self.transform = transform
        if patientCSV is not None:
            self.patientData = _loadPatientsCSV(patientCSV)
        else:
            self.patientData = None
        if lineageCSV is not None:
            self.lineageData = _loadLineageCSV(lineageCSV)
        else:
            self.lineageData = None
        patcher = Patcher(patchParams, patchRootDir)
        if patchType == "tumor":
            self.patchFiles = patcher.tumorPatches
        elif patchType == "neutral":
            self.patchFiles = patcher.neutralPatches
        else:
            raise RuntimeError(f"Invalid patch type: {patchType}")
        self.trainFiles, self.valFiles = self._trainValSplit(patcher.metadataFile)
        print(f"Total patch files: {len(self.patchFiles)}")
        print(f"Train files: {len(self.trainFiles)}")
        print(f"Val files: {len(self.valFiles)}")

    def _trainValSplit(self, metadataFile: str) -> Tuple[str, str]:
        with open(metadataFile) as f:
            metadata = yaml.load(f, Loader=yaml.BaseLoader)
        datasetRootDir = pathlib.Path(metadataFile).parent
        filesKey = f"{self.patchType}Files"
        if filesKey in metadata.keys():
            trainFiles = [
                datasetRootDir.joinpath(f) for f in metadata[filesKey]["trainFiles"]
            ]
            valFiles = [
                datasetRootDir.joinpath(f) for f in metadata[filesKey]["valFiles"]
            ]
        else:
            totalFiles = len(self.patchFiles)
            trainNFiles = int(self.TRAIN_VAL_RATIO * totalFiles)
            trainFiles = list(
                np.random.choice(self.patchFiles, trainNFiles, replace=False)
            )
            valFiles = list(set(self.patchFiles) - set(trainFiles))
            _trainFiles = [
                str(pathlib.Path(f).relative_to(pathlib.Path(f).parents[2]))
                for f in trainFiles
            ]
            _valFiles = [
                str(pathlib.Path(f).relative_to(pathlib.Path(f).parents[2]))
                for f in valFiles
            ]
            metadata.update(
                {filesKey: {"trainFiles": _trainFiles, "valFiles": _valFiles}}
            )
            with open(metadataFile, "w") as f:
                yaml.dump(metadata, f, Dumper=yaml.RoundTripDumper)
        return trainFiles, valFiles

    def __len__(self) -> int:
        if self.train:
            length = len(self.trainFiles)
        else:
            length = len(self.valFiles)
        return length

    def __extractLabelInfo__(self, patchFile: str) -> Dict:
        basename = pathlib.Path(patchFile).stem.split(".")[0]
        tumorInfo = basename.split("SR")[-1].split("-")[0]
        lineage = tumorInfo[-1]
        patientID = tumorInfo.split(lineage)[0]
        mouseInfo = basename.split("-")[1:]
        mouseID = mouseInfo[0]
        mouseSurvival = mouseInfo[1].split("_")[0]
        patchScore = basename.split("_")[1].split("_")[0]
        patchCoordsInfo = basename.split("_")[2:]
        x1, y1 = patchCoordsInfo[0], patchCoordsInfo[1]
        x2, y2 = patchCoordsInfo[2], patchCoordsInfo[3]
        patientLabel = alphabet.index(patientID)  # unique label for each patient
        lineageLabel = 26 * patientLabel + int(
            lineage
        )  # unique label for each cell lineage
        labels = dict(
            patientID=patientID,
            lineage=int(lineage),
            patientLabel=patientLabel,
            lineageLabel=lineageLabel,
            mouseID=mouseID,
            mouseSurvival=int(mouseSurvival),
            score=int(patchScore),
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            patchFile=str(patchFile),
        )
        if self.patientData is not None:
            patientDict = self.patientData[f"SR{patientID}"]
            labels.update(patientDict)
        if self.lineageData is not None:
            lineageData = self.lineageData[f"SR{patientID}{int(lineage)}"]
            labels.update(lineageData)
        return labels

    def __getitem__(self, imgIdx: int) -> Tuple[np.ndarray, Dict]:
        """Get a patch and its metadata

        Args:
            imgIdx (int): Patch index (used by dataloaders)

        Returns:
            Tuple[np.ndarray, Dict]: patch data,
            dictionnary containing:
            patient ID,
            tumor cell lineage,
            mouse ID,
            mouse survival,
            x,y coordinates of upper left patch corner
            x,y coordinates of lower right patch corner
        """
        if self.train:
            files = self.trainFiles
        else:
            files = self.valFiles
        if torch.is_tensor(imgIdx):
            imgIdx = imgIdx.tolist()
        patchFile = files[imgIdx]

        patchData = Image.open(patchFile).convert("RGB")
        if self.transform is not None:
            patchData = self.transform(patchData)
        labels = self.__extractLabelInfo__(patchFile)
        return patchData, labels


class TumorDataset(PatchDataset):
    def __init__(
        self,
        patchRootDir: str,
        patchParams: EasyDict,
        train: bool = True,
        transform: Callable = None,
        patientCSV: str = None,
        lineageCSV: str = None,
    ) -> None:
        """Tumor patches dataset constructor

        Args:
            patchRootDir (str): Path to folder containing patches
            patchParams (EasyDict): Patch extraction parameters
            train (bool): Train or Validation
            transform (Callable): Transforms to apply
            patientsCSV (str): Path to patients preprocessed CSV file
            radioResistanceCSV (str): Path to patients' radioresistance preprocessed CSV file
        """
        super().__init__(
            "tumor",
            patchRootDir,
            patchParams,
            train,
            transform,
            patientCSV,
            lineageCSV,
        )


class NoiseDataset(data.Dataset):
    def __init__(self) -> None:
        self.length = 1000
        self.image_size = 256
        self.channels = 3

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict:
        patient_id = random.randrange(10)
        random_image = torch.rand(self.channels, self.image_size, self.image_size)
        random_image = (random_image - random_image.min()) / (
            random_image.max() - random_image.min() + 1e-9
        )
        return random_image, patient_id
