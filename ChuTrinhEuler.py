import random
import copy
import timeit
import networkx as nx # type: ignore
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import pygame
import os
from PIL import Image, ImageTk

SET_COLOR = "\x1b[48;5;"
END = "\x1b[0m"
CLEAR = "[033[H"

RED = "#c6260d"
WHITE = "#ffffff"
BLACK = "#000000"
GREEN = "#22e11f"
YELLOW = "#fef65b"
BLUE = "#2a7cc1"

directory_path = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
icon_path = os.path.join(directory_path, 'icon.ico')
image_path = os.path.join(directory_path, 'image.png')
music_path = os.path.join(directory_path, 'Spin The Wheel.mp3')
original_image = Image.open(image_path)

class Graph:
    def __init__(self):
        # Init list adjacency list
        self.graph = {}

    def init_graph(self, edges):
        for u, v in edges:
            if u not in self.graph:
                self.graph[u] = []
            if v not in self.graph:
                self.graph[v] = []
            self.graph[u].append(v)
            self.graph[v].append(u)
    
    def remove_edge(self, u, v):
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if u in self.graph[v]:
            self.graph[v].remove(u)
        print(f"Graph after removing edge ({u}, {v}):", self.graph)
    
    def find_eulerian_cycle(self):
        # Check conditions exist of Eulerian cycle
        d=0
        a=[]
        for current in self.graph:
            if len(self.graph[current]) % 2 != 0:
                d+=1
                if d>2:
                    return "Graph not have Eulerian cycle" 
                a.append(current)
        # Do if it be Euler path
        if d==2:
            graph_temp = copy.deepcopy(self.graph)

            # Hierholzer
            stack = []
            cycle = []
            current  = random.choice(a)  # Choose random node!
            stack.append(current)

            while stack:
                if graph_temp[current]:                    # If node have edges not pass!
                    next_node = graph_temp[current][0]     # Get next_node randomly
                    stack.append(next_node)
                    
                    graph_temp[current].remove(next_node)
                    graph_temp[next_node].remove(current)  # Delete edge after pass
                    
                    current = next_node
                else:
                    current = stack.pop()
                    cycle.append(current)

            return cycle[::-1]


        # Make a copy
        graph_temp = copy.deepcopy(self.graph)

        # Hierholzer
        stack = []
        cycle = []
        current  = next(iter(self.graph))  # Choose random node!
        stack.append(current)

        while stack:
            if graph_temp[current]:                    # If node have edges not pass!
                next_node = graph_temp[current][0]     # Get next_node randomly
                stack.append(next_node)
                
                graph_temp[current].remove(next_node)
                graph_temp[next_node].remove(current)  # Delete edge after pass
                
                current = next_node
            else:
                current = stack.pop()
                cycle.append(current)

        if cycle[0] != cycle[-1]:
            cycle.append(cycle[0])
        return cycle[::-1]
       
def generate_adjacency_list(vertex):
    edges = set()
    n = vertex*(vertex - 1)//2
    # n =random.randint(5, vertex*(vertex - 1)//2 )
    while len(edges) < n:  # Ensure no overnumbers edge create!
        u = random.randint(1, vertex)
        v = random.randint(1, vertex)
        if u != v:
            edges.add(tuple(sorted((u, v))))  # sorted to envade leap
        # else:
        #     n-=1
    return edges   

def draw_graph_with_euler_cycle(graph, euler_cycle):
    # Init graph net
    G = nx.Graph()
    P = nx.DiGraph()
    # Add edges to graph
    for node, adjacents in graph.graph.items():
        for adj in adjacents:
            if (adj, node) not in G.edges():
                G.add_edge(node, adj)
                P.add_edge(node, adj)

    plt.close()

    pos = nx.spring_layout(G)   # Bố cục
    # pos2 = nx.spring_layout(P)
    
    # Create subplots to show two graphs in one window
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Draw basic Graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size = 700, font_size=10, ax=ax1)
    ax1.set_title("Original Graph")

    # Draw Euler Cycle
    if isinstance(euler_cycle, list):
        if euler_cycle[0] == euler_cycle[-1]:
            print("Cycle Euler Found: ", euler_cycle)
            ax2.set_title("Graph with Euler Cycle")
            # Add List EulerEuler
            max_line_length = 105  # Max characters per line
            euler_cycle_str = "Euler Cycle: " + " -> ".join(map(str, euler_cycle))
        else:
            print("Euler path Found: ", euler_cycle)
            ax2.set_title("Graph with Euler path")
            # Add List EulerEuler
            max_line_length = 105  # Max characters per line
            euler_cycle_str = "Euler path: " + " -> ".join(map(str, euler_cycle))    


        euler_edges = [(euler_cycle[i], euler_cycle[i+1]) for i in range(len(euler_cycle) - 1)]
        nx.draw_networkx_edges(P, pos, edgelist = euler_edges, edge_color='r', width=2, ax=ax2, arrows=True, arrowsize=40)
            

        # Add node and labels
        nx.draw_networkx_nodes(P, pos, node_color='lightblue', node_size=700, ax=ax2)
        nx.draw_networkx_labels(P, pos, font_size=10, ax=ax2)

        
        wrapped_text = "\n".join([euler_cycle_str[i:i+max_line_length] for i in range(0, len(euler_cycle_str), max_line_length)])
        ax2.text(0.5, -0.15, wrapped_text, ha='center', va='center', fontsize=10, transform=ax2.transAxes)
    else:
        ax2.set_title("Graph does not have Eulerian cycle")
        print("Graph not have Cycle Euler")
    # valid_euler_edges = [edge for edge in euler_edges if edge in G.edges()]
    # if valid_euler_edges:
    

    plt.tight_layout()
    plt.show()

def run_graph():
    vertex = int(entry_vertex.get())
    edges = generate_adjacency_list(vertex)
    graph = Graph()
    graph.init_graph(edges)
    
    print ("Adjacency list of Graph:")
    for node, adjacents in graph.graph.items():
        print(f'Vertex {node}: {adjacents}')
    
    cycle = graph.find_eulerian_cycle()
    # Draw
    draw_graph_with_euler_cycle(graph, cycle)
    
    # Measure time play
    execution_time = timeit.timeit(lambda: graph.find_eulerian_cycle(), number=1)
    print(f"Time taken to find Eulerian cycle: {execution_time} seconds")

# Init GUI for user!!
def resize_image(event):
    new_width = event.width
    new_height = event.height
    resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    root.bg_img = ImageTk.PhotoImage(resized_image)
    label.config(image = root.bg_img)

def clear_graph():
    # Clear the graph
    plt.close()  # Clear the current figure
    print("Graph cleared!")

def init_GUI():
    global root, label
    root =  tk.Tk()
    root.title("Eulerian Cycle Finder")
    root.geometry("1000x800")

    #set icon
    root.iconbitmap(icon_path)

    #set background
    root.bg_img = ImageTk.PhotoImage(file = image_path)
    label = tk.Label(root, image = root.bg_img)
    label.pack(fill='both', expand = True)
    
    root.bind('<Configure>', resize_image)

    frame = tk.Frame(root, bg=BLUE, width=300, height=100)
    frame.place(relx=0.65, rely=1, anchor='s', y=-20)

    # Create input fields and labels
    label_vertex = tk.Label(frame, text = "Enter number of vertices:")
    label_vertex.pack(pady=5)

    global entry_vertex
    entry_vertex = tk.Entry(frame, justify="center")
    entry_vertex.pack(pady=5)

    run_button = tk.Button(frame, text = "Generate Graph & Find Euler Cycle", bg=YELLOW , command=run_graph)
    run_button.pack(pady=10)

    clear_button = tk.Button(frame, text='CLEAR',width=8, bg = RED,  command=lambda: clear_graph())
    clear_button.pack(pady=5)

def exit(music, root):
    music.stop()
    root.destroy()


if __name__ == "__main__":
    init_GUI()



    # Initialize Pygame mixer
    pygame.mixer.init()
    # Load and play background music 
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play(-1)  # -1 means loop indefinitely

    root.mainloop()
    
    