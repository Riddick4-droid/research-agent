from graphviz import Digraph
#from utils.logger import get_logger


def visualize_graph():
    #logger = get_logger('visualize_graph')

    dot = Digraph(comment="Reseaarch Agent Graph")

    dot.node('Planner')
    dot.node('Human Review')
    dot.node('Research')
    dot.node('Summarize')
    dot.node('Memory Update')
    dot.node('Synthesis')
    dot.node('Writer')

    dot.edge("planner","Human Review")
    dot.edge("Human Review","Research")

    dot.edge("Research","Summarize")
    dot.edge("Summarize","Memory Update")

    dot.edge("Memory Update","Research",label='next subtopic')

    dot.edge("Memory Update","Synthesis", label='done')
    
    dot.edge("Synthesis","Writer")

    return dot

if __name__ == '__main__':
    graph = visualize_graph()
    graph.render('research_grpah',format='png',view=True)


