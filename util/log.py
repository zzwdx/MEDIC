def log(str, path):
    print(str)
    with open(path, 'a') as f: 
        f.write(str + '\n') 

def save_data(data, path):
    with open(path, 'a') as f: 
        f.write(str(data) + '\n') 