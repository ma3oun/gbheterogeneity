from typing import List, Tuple
import cv2
import os, pathlib
from easydict import EasyDict
from typing import List
import numpy as np
import yaml as yaml


def __coord2str__(coords: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
    """Transform patch coordinates to text

    Args:
        coords (Tuple[Tuple[int, int], Tuple[int, int]]): Patch image coordinates (upper left XY, lower right XY)

    Returns:
        str: Patch coordinates to be put into the patch filename
    """
    s = f"{coords[0][0]:05d}_{coords[0][1]:05d}_{coords[1][0]:05d}_{coords[1][1]:05d}"
    return s

class Patch:
    def __init__(
        self,
        patchData: np.ndarray,
        coordinates: Tuple[Tuple[int, int], Tuple[int, int]],
        score: Tuple[int, int],
        imgFile: str,
    ) -> None:
        """Patch object constructor

        Args:
            patchData (np.ndarray): Patch BGR data
            coordinates (Tuple[Tuple[int, int], Tuple[int, int]]): Patch image coordinates (upper left XY, lower right XY)
            score (Tuple[int, int]): Tumor score, Neutral score
            imgFile (str): Path to full image file the patch was extracted from
        """
        self.data = patchData
        self.coords = coordinates
        self.tumorScore = score[0]
        self.neutralScore = score[1]
        self.tumorIdx = None
        self.neutralIdx = None
        self.imgFile = imgFile

    def save(self, outputDir: str, patchType: str = "tumor"):
        """Saves the patch into the output directory, within the subdirectory corresponding to its type.
        The folder structure is: outputDir |--image1---|---neutral
                                           |           |---tumor
                                           |
                                           |--image2---|---neutral
                                           |           |---tumor

        Args:
            outputDir (str): Output directory where patches will be stored
            patchType (str, optional): "tumor" or "neutral". Defaults to "tumor".
        """
        strippedFilename = pathlib.Path(self.imgFile).stem
        strippedFilename = strippedFilename.split("_")[0]  # remove magnification value
        if patchType == "tumor":
            filename = f"{strippedFilename}_{self.tumorIdx:05d}_{__coord2str__(self.coords)}.png"
        else:
            filename = f"{strippedFilename}_{self.neutralIdx:05d}_{__coord2str__(self.coords)}.png"
        finalOutputDir = os.path.join(outputDir, strippedFilename, patchType)
        fullPath = os.path.join(finalOutputDir, filename)
        os.makedirs(finalOutputDir, exist_ok=True)
        cv2.imwrite(fullPath, self.data)
        return

class Patcher:
    def __init__(
        self, patchParams: EasyDict, outputDir: str, rootDir: str = None
    ) -> None:
        if rootDir is not None:
            self.root = self.__get_root__(rootDir)
        else:
            print("No root directory provided. Assuming patches have already been extracted.")
            self.root = None
        self.outputDir = outputDir

        if not isinstance(patchParams, EasyDict):
            patchParams = EasyDict(patchParams)

        for attr in [
            "patchH",
            "patchW",
            "sequentialSampling",
            "tumorColor",
            "neutralColor",
            "colorMargin",
            "minS",
            "maxS",
            "minL",
            "maxL",
            "tumorMinScore",
            "neutralMinScore",
            "maxPatchesPerImage",
        ]:
            setattr(self, attr, getattr(patchParams, attr))

        self.metadataFile = os.path.join(self.outputDir, "metadata.yml")

    def __getPatchFiles__(self, patchType: str):
        if os.path.exists(self.metadataFile):
            print(
                f"Found metadata file. Assuming patches have already been extracted."
            )
        else:
            raise RuntimeError(
                f"Metadata file not found. Make sure data has been downloaded at the specified folder"
            )
        folderList = [
            os.path.join(self.outputDir, f)
            for f in os.listdir(self.outputDir)
            if os.path.isdir(os.path.join(self.outputDir, f))
        ]
        patchFiles = []
        for folder in folderList:
            subDir = os.path.join(folder, patchType)
            try:
                patches = [
                    os.path.join(subDir, p)
                    for p in os.listdir(subDir)
                    if os.path.isfile(os.path.join(subDir, p))
                ]
            except FileNotFoundError:
                continue
            patchFiles.extend(patches)
        return patchFiles

    @property
    def tumorPatches(self):
        return self.__getPatchFiles__("tumor")

    @property
    def neutralPatches(self):
        return self.__getPatchFiles__("neutral")

    def getRawFilesList(self) -> List[str]:
        """Build the image file list from the patches root directory

        Returns:
            List[str]: List of absolute paths to full size images
        """
        fileList = [os.path.join(self.root, f) for f in os.listdir(self.root)]
        fileList = [f for f in fileList if os.path.isfile(f)]
        return fileList

    def __get_root__(self, rootDir: str) -> str:
        """Get the patches root directory, containing all full size images to consider

        Args:
            rootDir (str): Path to root directory

        Returns:
            str: Patches root directory
        """

        if not os.path.isdir(rootDir):
            raise FileNotFoundError("Patch data directory does not exist")
        return rootDir
