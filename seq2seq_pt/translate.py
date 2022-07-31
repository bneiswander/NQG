from __future__ import division

import logging
import math
import time
from types import SimpleNamespace

import torch

import constants
import s2s

opt_translate = {
    "model": f"{SAVEPATH}/model_e20.pt",
    "src": f"{DATAHOME}/test_sample/dev.txt.shuffle.test.source.txt",
    "bio": f"{DATAHOME}/test_sample/dev.txt.shuffle.test.bio",
    "feats": [
        f"{DATAHOME}/test_sample/dev.txt.shuffle.test.case",
        f"{DATAHOME}/test_sample/dev.txt.shuffle.test.ner",
        f"{DATAHOME}/test_sample/dev.txt.shuffle.test.pos",
    ],
    "tgt": f"{DATAHOME}/test_sample/dev.txt.shuffle.test.target.txt",
    "output": f"{DATAHOME}/test_sample/dev.txt.shuffle.test.predictions.txt",
    "beam_size": 12,
    "batch_size": 64,
    "max_sent_length": 100,
    "replace_unk": True,
    "verbose": True,
    "n_best": 10,
    "gpu": 0,
}

opt_translate = SimpleNamespace(**opt_translate)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s:%(name)s]: %(message)s", level=logging.INFO
)
file_handler = logging.FileHandler(
    time.strftime("%Y%m%d-%H%M%S") + ".log.txt", encoding="utf-8"
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s")
)
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)


def reportScore(name, scoreTotal, wordsTotal):
    logger.info(
        "%s AVG SCORE: %.4f, %s PPL: %.4f"
        % (name, scoreTotal / wordsTotal, name, math.exp(-scoreTotal / wordsTotal))
    )


def addone(f):
    for line in f:
        yield line
    yield None


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)


def translateMain(opt):
    logger.info(opt)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = s2s.Translator(opt)

    outF = open(opt.output, "w", encoding="utf-8")

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []
    bio_batch, feats_batch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    bioF = open(opt.bio, encoding="utf-8")
    featFs = [open(x, encoding="utf-8") for x in opt.feats]
    for line in addone(open(opt.src, encoding="utf-8")):

        if line is not None:
            srcTokens = line.strip().split(" ")
            srcBatch += [srcTokens]
            bio_tokens = bioF.readline().strip().split(" ")
            bio_batch += [bio_tokens]
            feats_tokens = [reader.readline().strip().split((" ")) for reader in featFs]
            feats_batch += [feats_tokens]
            if tgtF:
                tgtTokens = tgtF.readline().split(" ") if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        predBatch, predScore, goldScore = translator.translate(
            srcBatch, bio_batch, feats_batch, tgtBatch
        )

        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        # if tgtF is not None:
        #     goldScoreTotal += sum(goldScore)
        #     goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write(" ".join(predBatch[b][0]) + "\n")
            outF.flush()

            if opt.verbose:
                srcSent = " ".join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                logger.info("SENT %d: %s" % (count, srcSent))
                logger.info("PRED %d: %s" % (count, " ".join(predBatch[b][0])))
                logger.info("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = " ".join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    logger.info("GOLD %d: %s " % (count, tgtSent))
                    # logger.info("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    logger.info("\nBEST HYP:")
                    for n in range(opt.n_best):
                        logger.info(
                            "[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n]))
                        )

                logger.info("")

        srcBatch, tgtBatch = [], []
        bio_batch, feats_batch = [], []

    reportScore("PRED", predScoreTotal, predWordsTotal)
    # if tgtF:
    #     reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    logger.info("{0} copy".format(translator.copyCount))
