
from exp import AttrDict
import model

def loadModel(filePrefix):
    jsonFile = filePrefix + ".json"
    weightFile = filePrefix + ".h5"
    jFile = open(jsonFile, 'r')
    loaded_model_json = jFile.read()
    jFile.close()
    mod = model_from_json(loaded_model_json)
    mod.load_weights(weightFile)
    print("Loaded model from files {}, {}".format(jsonFile, weightFile))
    return mod

def saveModel(mod, filePrefix):
    weightFile = filePrefix + ".h5"
    mod.save_weights(weightFile)
    jsonFile = filePrefix + ".json"
    with open(filePrefix + ".json", "w") as json_file:
        try:
            json_file.write(mod.to_json())
        except:
            print("Failed saving json file {}.json, you have to load this file using load_models.rebuild_models(args)".format(filePrefix))
        else:
            print("Saved model to files {}, {}".format(jsonFile, weightFile))


def load_autoencoder(args):
    prefix = args.prefix
    try:
        modelDict = AttrDict({})
        modelDict.ae = loadModel(prefix + "model")
        modelDict.encoder = loadModel(prefix + "_encoder")
        modelDict.encoder_var = loadModel(prefix + "_encoder_var")
        modelDict.generator = loadModel(prefix + "_generator")
        if args.decoder == "gaussian":
            modelDict.generator_mixture = loadModel(prefix + "_generator_mixture")
    except:
        modelDict = rebuild_models(args)

    return modelDict

# return (autoencoder, encoder, encoder_var, generator)
def rebuild_models(args):
    modelDict = model.build_model(args)
    for key in list(modelDict.keys()):
        if key == "ae":
            curr_model_name = "model"
        else:
            curr_model_name = key
        curr_model = modelDict[key]
        weightFile = args.prefix + "_" + curr_model_name + ".h5"
        curr_model.summary()
        print("Loading weights from file: ", weightFile)
        try:
            curr_model.load_weights(weightFile)
        except:
            print("Failed to load weights from file {}".format(weightFile))
    return modelDict

