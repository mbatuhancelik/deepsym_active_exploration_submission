import argparse
import torch
from threading import Thread
import pickle

from council_manager import download_experiment, load_generation
from dataset import StateActionEffectDataset
import astar

def evaluate_model_on_dataset(model, model_name, dataset_name, result_map):
    astar.model = model
    dataset = StateActionEffectDataset(dataset_name, "test")
    result_map[(model_name[0],model_name[1],dataset_name)] = astar.evaluate_dataset(model, dataset, max_depth= 1)
def evaluate_model_on_collection_datasets(model,model_name, data_experiment_name, num_generations, result_map):
    for i in range(num_generations):
        evaluate_model_on_dataset(dataset_name=f"{data_experiment_name}_collection_{i}",
                                result_map=result_map,
                                model=model,
                                model_name=model_name
                                )
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser("See dataset metrics.")
    parser.add_argument("-m", help="model experiment name", type=str)
    parser.add_argument("-d", help="dataset_name", type=str)
    parser.add_argument("-i", help="num_generations", type=int)
    parser.add_argument("-init", help="initial dataset", type=str)

    args = parser.parse_args()

    args.i = args.i + 1
    
    result_map = {}
    for i in range(args.i):
        print(f"gen {i + 3} processing")
        models = load_generation(i+3, args.m)
        threads = []
        for k, model in enumerate(models):
            # evaluate_model_on_dataset(model, (i,k), args.init, result_map)
            evaluate_model_on_collection_datasets(model, (i,k), args.d, args.i, result_map)
    
    with open(f'{args.m}_on_{args.d}.pkl', 'wb') as f:  # open a text file
        pickle.dump(result_map, f) # serialize the list
        





    


    
