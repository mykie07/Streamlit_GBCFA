import streamlit as st
import streamlit.components.v1 as components
# import matplotlib.pyplot as plt

import py2neo
from py2neo import Node, Relationship, Graph, NodeMatcher
# import altair as alt
import pandas as pd

# import networkx as nx
# from pyvis.network import Network
import json

import plotly.graph_objects as go

# from pyecharts.charts import Bar
# from pyecharts import options as opts

#connect to neo4j
#Specifically GBCFA (Graph-based Customer Feedback Analaysis) database
G= Graph("bolt://localhost:7687",password="gbcfa")

# st.beta_set_page_config(layout="wide")
# st.beta_set_page_config(layout="wide")
st.title("A dashboard for flexible analysis of customer feedback. NEO4j + Streamlit")
st.header("Aspect-Opinion analysis of Samsung Chromebook 12 reviews")

intro_options= st.selectbox(
    "Show introduction?",
    ["NO","YES"]

)

if intro_options=="YES":
    st.write("This dashboard is powered by the labeled property graph model illustrated below. Each customer feedback/review in the textual format is made up of SENTENCES that in turn express opinion on various product/service aspect keywords. These aspect classes belong to specific CLASSES being tracked by domain experts. In an earlier processing step, the data is preprocessed and aspects are tagged with sentiment scores. The resulting dataset is then modeled in the Neo4j graph database.")

    st.image("FIG4_graph_model_schema.jpg")

# limit = st.slider("Choose your weight", min_value=5, max_value=20, step=1)



#GET PRODUCT ASPECT CLASS IDs
class_IDs =  G.run('''
MATCH  (class:Class) return ID(class) as id, class.name as class
 ''').data()

class_df = pd.DataFrame(class_IDs)
# id_list =list(class_df["id"])
aspect_classes = list(class_df["class"])

# index=[]
# for t in list(top_terms):
#     index.append(t['token'])

# st.write(index)

class_options = st.multiselect(
     'Which of the aspect classes do want to include in the general summary?',
     aspect_classes,
     aspect_classes)

selected_classes_df = class_df.loc[class_df['class'].isin(class_options)]
# top_terms_tokens=list(selected_df["token"])
selected_classes_ids=list(selected_classes_df["id"])

# st.write(selected_classes_ids)

#Generate Opinion Statistics for the 17 aspect classes of the product
aspect_class_opinion_stats=[]
for ID in selected_classes_ids:
    classid=ID
    all_opinions = G.run('''
    MATCH (class:Class)
    WHERE id(class)=$classid
    MATCH (class)<-[:BELONGS_TO]-(key1:Keyword)<-[o1:HAS_OPINION_ON]-(sent1:Sentence)<-[:HAS_SENTENCE]-(rev:Review)
    //count(o1) as all_counts
    RETURN class.name as aspect_class, count(o1) as all_counts
    ''',parameters={'classid':classid}).data()

    neg_opinions = G.run('''
    MATCH (class:Class)
    WHERE id(class)=$classid
    MATCH (class)<-[:BELONGS_TO]-(key1:Keyword)<-[o1:HAS_OPINION_ON]-(sent1:Sentence)<-[:HAS_SENTENCE]-(rev:Review)
    WHERE o1.polarity<0 
    RETURN count(o1) as neg_counts
    ''',parameters={'classid':classid}).data()

    pos_opinions = G.run('''
    MATCH (class:Class)
    WHERE id(class)=$classid
    MATCH 
    (class)<-[:BELONGS_TO]-(key1:Keyword)<-[o1:HAS_OPINION_ON]-(sent1:Sentence)<-[:HAS_SENTENCE]-(rev:Review)
    WHERE o1.polarity>0 
    RETURN count(o1) as pos_counts
    ''',parameters={'classid':classid}).data()

    neu_opinions = G.run('''MATCH (class:Class)
    WHERE id(class)=$classid
    MATCH (class)<-[:BELONGS_TO]-(key1:Keyword)<-[o1:HAS_OPINION_ON]-(sent1:Sentence)<-[:HAS_SENTENCE]-(rev:Review)
    WHERE o1.polarity=0
    RETURN count(o1) as neu_counts
    ''',parameters={'classid':classid}).data()
    
    
    aspect_class_opinion_stats.append([all_opinions[0]['aspect_class'], all_opinions[0]['all_counts'],pos_opinions[0]['pos_counts'],pos_opinions[0]['pos_counts']/all_opinions[0]['all_counts']*100,neg_opinions[0]['neg_counts'],neg_opinions[0]['neg_counts']/all_opinions[0]['all_counts']*100, neu_opinions[0]['neu_counts'], neu_opinions[0]['neu_counts']/all_opinions[0]['all_counts']*100]
)

ABSAPI_stats = pd.DataFrame(aspect_class_opinion_stats,columns=["aspect_class","all_counts","pos_counts","percent_pos","neg_counts","percent_neg","neu_counts","percent_neu"])



st.write(ABSAPI_stats.sort_values(['all_counts','percent_neg'], ascending=False))


data=[ go.Pie(labels=ABSAPI_stats['aspect_class'],
           values=ABSAPI_stats['all_counts'],
           hole=.7)]
fig = go.Figure(data = data)
st.plotly_chart(fig)



datadf= ABSAPI_stats.sort_values(['all_counts'], ascending=False)
datadf= datadf[['aspect_class','all_counts','neu_counts','pos_counts', 'neg_counts']]
X=list(datadf["aspect_class"])

fig2 = go.Figure() 
fig2.add_trace(go.Bar(x=X, y=datadf['pos_counts'], name="positive",marker_color='indianred'))
fig2.add_trace(go.Bar(x=X, y=datadf['neg_counts'], name="negative",marker_color='lightsalmon'))
fig2.add_trace(go.Bar(x=X, y=datadf['neu_counts'], name="neutral",marker_color='sandybrown'))
fig2.update_layout(barmode="stack")
fig2.update_layout(
    title="Breakdown of sentiment counts per aspect class",
    xaxis_title="Aspect keywords",
    yaxis_title="Sentiment count",
    legend_title="Sentiment classes",
    font=dict(
    #     family="Courier New, monospace",
        size=15,
        color="purple"
    )
    )
st.write(fig2)




#Co-mention analysis
st.subheader("Co-mention analysis")

className = st.selectbox(
     'Which aspect class would you like to focus on',
     aspect_classes)

neg_co_mention_query='''
match (class1)<-[:BELONGS_TO]-(key1:Keyword)<-[o1:HAS_OPINION_ON]-(sent:Sentence)-[o2:HAS_OPINION_ON]->(key2:Keyword)-[:BELONGS_TO]->(class2)
where class1.name=$className and o2.polarity>0
//RETURN class2.name,key2.keyword,collect(o2.polarity) order by class2.name
RETURN class2.name as class,key2.keyword as keyword,o2.polarity as polarity,sent.text as sentence order by class2.name
'''

pos_co_mention_query='''
match (class1)<-[:BELONGS_TO]-(key1:Keyword)<-[o1:HAS_OPINION_ON]-(sent:Sentence)-[o2:HAS_OPINION_ON]->(key2:Keyword)-[:BELONGS_TO]->(class2)
where class1.name=$className and o2.polarity<0
//RETURN class2.name,key2.keyword,collect(o2.polarity) order by class2.name
RETURN class2.name as class,key2.keyword as keyword,o2.polarity as polarity,sent.text as sentence order by class2.name
'''

aspects_pos_commentions = G.run(neg_co_mention_query,parameters={'className':className}).data()
aspects_pos_commentions_df = pd.DataFrame(aspects_pos_commentions)
# st.write(aspects_po_commentions)
figcoNeg = go.Figure(data=go.Scatter(
            x=aspects_pos_commentions_df['keyword'],
            y=aspects_pos_commentions_df['polarity'],
            mode='markers',
            marker_size=6,
            marker_color=aspects_pos_commentions_df['polarity'],
            # showscale=True,
            # colorscale='Viridis',
            #,
            text=aspects_pos_commentions_df['sentence'],
           
            )
            # 
            ) # hover text goes here
figcoNeg.update_layout(
    title="Plot of positive aspect-opinions co-mentioned with ....." + className,
    xaxis_title="Aspect keywords",
    yaxis_title="Sentiment score",
    legend_title="Legend Title",
    font=dict(
        # family="Courier New, monospace",
        size=15,
        # color="RebeccaPurple"
        color="purple"
    )
    )
        # fig.update_layout(title='Filtered summary')
st.write(figcoNeg)

aspects_neg_commentions = G.run(pos_co_mention_query,parameters={'className':className}).data()
aspects_neg_commentions_df = pd.DataFrame(aspects_neg_commentions)
# st.write(aspects_po_commentions)
figcoPos = go.Figure(data=go.Scatter(
            x=aspects_neg_commentions_df['keyword'],
            y=aspects_neg_commentions_df['polarity'],
            mode='markers',
            marker_size=6,
            marker_color=aspects_neg_commentions_df['polarity'],
            # showscale=True,
            # colorscale='Viridis',
            #,
            text=aspects_neg_commentions_df['sentence']
            )
            # 
            ) # hover text goes here

figcoPos.update_layout(
    title="Plot of positive aspect-opinions co-mentioned with ....." + className,
    xaxis_title="Aspect keywords",
    yaxis_title="Sentiment score",
    legend_title="Legend Title",
    font=dict(
    #     family="Courier New, monospace",
        size=15,
        color="purple"
    )
    )
        # fig.update_layout(title='Filtered summary')
st.write(figcoPos)






#SEARCH BY INPUT TEXT: GET SENTENCES BASED ON FULLTEXT INDEX ON SENTENCES
st.title("Combine fulltext search with product aspect-based sentiment filtering")
user_input = st.text_input(label="Enter search phrase")

if user_input:
    index_search = G.run('''
    CALL db.index.fulltext.queryNodes("sentenceIndex", $user_input) YIELD node, score
    RETURN id(node) as id, node.text as text, score
    ''',parameters={'user_input':user_input}).data()

    # if index_search

    if index_search:
        df=pd.DataFrame(index_search)
        # st.write(df)
        # text = " ".join(sentence for sentence in df.text)
        # st.write(pd.DataFrame(index_search))
        # st.write(text)

        search_idList= list(df['id'])
        st.write("Number of sentences:",len(search_idList))
        # st.table(df)

        tablefig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
                cells=dict(values=[df.id, df.text, df.score],
                fill_color='lavender',
                align='left'))
                ])

        st.write(tablefig)
    

        # Query0='''
        # MATCH (sent1:Sentence)-[o1:HAS_OPINION_ON]->(key:Keyword)-[:BELONGS_TO]->(class)
        # WHERE sent1.number in $search_idList
        # RETURN distinct( class.name) AS class, distinct(key.keyword) as keyword
        # '''
        classKeywordQuery='''
        MATCH (sent1:Sentence)-[o1:HAS_OPINION_ON]->(keys:Keyword)-[:BELONGS_TO]->(class)
        //where sent1.number in $search_idList
        where id(sent1) in $search_idList
        with class.name as class, collect (keys.keyword) as keyword
        return class, keyword
        '''

        classKeywords=G.run(classKeywordQuery,parameters={'search_idList':search_idList}).data()
        if not(classKeywords):
            print("Sorry.....No keyword Match")
        # st.write(list(associated_classes))
        else:
            # st.write(classKeywords)
            classKeywords_df=pd.DataFrame(classKeywords)



            # matched_classes = list(pd.DataFrame(associated_classes)["class"])
            matched_classes = list(classKeywords_df["class"])
            # st.write(matched_classes)
            

            class_choices = st.multiselect(
            'Which of the matched classes would you like to include in the analysis?',
            matched_classes,
            matched_classes)

            #REFRESH THE KEYWORDS LIST ACCORDING TO THE ACTIVE CLASS CHOICES
            #THIS CAN BE DONE IN PANDAS INSTEAD TAKING A TRIP TO THE DATABASE
            
            # matched_keywords = list(pd.DataFrame(associated_classes)["keyword"])
            filtered_ClassKeyword_df= classKeywords_df.loc[classKeywords_df['class'].isin(class_choices)]
            matched_keywords = filtered_ClassKeyword_df["keyword"]
            # st.write(pd.DataFrame(matched_keywords))
            sorted_keywords=[]
            for keyList in matched_keywords:
                class_keywords=[]
                for item in keyList:
                    class_keywords.append(item)
                # class_keywords.append(list(set(l)))
                sorted_keywords=sorted_keywords + class_keywords
            st.write("Counts of keyword mentions: ", len(sorted_keywords))
            sorted_keywords=sorted(list(set(sorted_keywords)))
            st.write("Count of sorted disctinct keywords: ",len(sorted_keywords))

            colL, colR = st.beta_columns(2)
            import copy
            sorted_keywordsNEG=copy.copy(sorted_keywords)
            with colL:
                keyword_choices = st.multiselect(
                'Which corresponding keywords of the matched classes would you like to include in the analysis?',
                sorted_keywords,
                sorted_keywords)
            with colR:
                keyword_choicesNEG = st.multiselect(
                'Which corresponding keywords of the matched classes would you like to include in the analysis?',
                sorted_keywordsNEG,
                sorted_keywordsNEG,key="NEG")
            # stats=[]
            # for classid in selected_classes_ids:
                # st.write(selected_classes_ids)

        Query_keyword_pos_stat ='''
        MATCH (sent1:Sentence)-[o1:HAS_OPINION_ON]->(KW:Keyword)-[:BELONGS_TO]->(class)
        WHERE id(sent1) IN $search_idList and class.name IN $class_choices and KW.keyword IN $keyword_choices 
        and (o1.polarity<>0.0)
        RETURN class.name as class, o1.polarity as polarity        
        '''
        filtered_pos = G.run(Query_keyword_pos_stat,parameters={'class_choices':class_choices,
        'keyword_choices':keyword_choices,
        'search_idList':search_idList,
        'classid':classid}).data()


        filtered_df = pd.DataFrame(filtered_pos)
        st.write(pd.DataFrame(filtered_pos).shape)
        # st.write(pd.DataFrame(filtered_pos))


        filtered_pos_df = filtered_df.loc[filtered_df['polarity']>0.0]
        # st.write(filtered_pos_df)

        filtered_neg_df = filtered_df.loc[filtered_df['polarity']<0.0]
        # st.write(filtered_neg_df)

        filtered_summarized_df = filtered_pos_df.groupby(["class"]).size().reset_index(name='pos_counts')
        # st.write(filtered_summarized_df)

        import numpy as np
        fig_fulltext_pos = go.Figure(data=go.Scatter(
            x=filtered_summarized_df['class'],
                y = filtered_summarized_df["pos_counts"],
                mode='markers',
                marker=dict(
                size=16,
                color=filtered_summarized_df["pos_counts"], #set color equal to a variable
                colorscale='Viridis', # one of plotly colorscales
                showscale=True
                # ,
                # text=filtered_summarized_df["class"]
                ),
                text=filtered_summarized_df["class"]

                ))
        fig_fulltext_pos.update_layout(
            title="Summary of all aspect classes associated with the search phrase :" +  user_input ,
            xaxis_title="Aspect keywords",
            yaxis_title="Sentiment count",
            legend_title="Legend Title",
            font=dict(
            #     family="Courier New, monospace",
                size=15,
                color="purple"
            )
            )
        st.write(fig_fulltext_pos)


        # fig1 = go.Figure(data=go.Scatter(
        #     x=filtered_df['class'],
        #     y=filtered_df['polarity'],
        #     mode='markers',
        #     marker_size=12,
        #     marker_color=filtered_df['polarity'],
        #     # showscale=True,
        #     # colorscale='Viridis',
        #     #,
        #     text=filtered_summarized_df['class']
        #     )
        #     # 
        #     ) # hover text goes here

        # # fig.update_layout(title='Filtered summary')
        # st.write(fig1)

            
    else:
        st.write("No records found. Please rephrase search. Thank you!!")

