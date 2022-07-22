########## Dependencies
import numpy as np
import pandas as pd
from scipy.stats import norm

import mysql.connector
from mysql.connector import Error

import altair as alt

import streamlit as st

sqlserver = st.secrets["sqlserver"]
sqlport = st.secrets["sqlport"]
sqluser = st.secrets["sqluser"]
sqlpwd = st.secrets["sqlpwd"]

########## SQL functions
def create_connection(host_name, port_no, user_name, user_password,dbname):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            port=port_no,
            user=user_name,
            passwd=user_password,
            database=dbname
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        st.write(f"The error '{e}' occurred")

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


def df_to_sql_insert(df_in,indexer=True,startval=0):
    indval = 1*startval
    data_for_query = ""
    for j in range(len(df_in)):
        curline = ""
        for element in ['"'*int(not str(element).replace('.','',1).isnumeric()) + str(element) + '"'*int(not str(element).replace('.','',1).isnumeric())  + "," for element in np.array(df_in.loc[df_in.index[j]])]:
            curline += element
        data_for_query += "(" 
        if indexer:
            data_for_query += str(indval) + ","
            indval += 1
        data_for_query += curline[0:-1] + ")," + "\n"
    return data_for_query[0:-2]

def get_current_max_ind(connect,table,indname,execute_read_query=execute_read_query):
    query = "SELECT MAX(" + indname + ") FROM " + table + ";"
    curmaxind = execute_read_query(connection,query)[0][0]
    if curmaxind is None:
        curmaxind = -1
    return curmaxind


@st.cache
def load_from_sql(sqlquery,sqlcreds,columnnames,execute_read_query=execute_read_query,create_connection=create_connection):
    sqlconnection = create_connection(*sqlcreds)
    datainput = execute_read_query(sqlconnection,sqlquery)
    sqlconnection.close()
    return pd.DataFrame(data=datainput,columns=columnnames)

highlighter = alt.selection_single(on='mouseover')


########## Custom visualization functions
def compare_and_chart_unbounded(df,varin,charttitle,legtitle,axtitle,highlighter=highlighter):
    mean1 = float(df.loc[0,(varin,"mean")])
    mean2 = float(df.loc[1,(varin,"mean")])

    pval = norm.cdf(1-abs(mean1 - mean2)
    /np.sqrt((float(df.loc[0,(varin,"std")])**2)/float(df.loc[0,("N","count")]) + (float(df.loc[1,(varin,"std")])**2)/float(df.loc[1,("N","count")]) )  )
    
    df_tochart = df[[('renewed',''),('N','count'),(varin,'mean')]].droplevel(1,axis=1)


    df_tochart["cur_std"] = df[varin]["std"]/np.sqrt(df["N"]["count"])
    df_tochart["lbound95"] = df[varin]["mean"] - 1.961*df_tochart["cur_std"]
    df_tochart["ubound95"] = df[varin]["mean"] + 1.961*df_tochart["cur_std"]


    curchart = alt.Chart(df_tochart,width=250,height=250,title=[charttitle,f"p-value: {pval:.3f}","95% C.I. in red"])
    curchart_bar = curchart.mark_bar().encode(x=alt.X("renewed:O"),y=alt.Y(varin + ":Q",scale=alt.Scale(),axis=alt.Axis(title=axtitle)),
                                                                    opacity=alt.condition(~highlighter,alt.value(0.9),alt.value(0.5)),
                                                                    tooltip=[alt.Tooltip(varin,title=legtitle,format=".1f"),
                                                                             alt.Tooltip("cur_std",title="Std. Dev.",format=".1f"),
                                                                             alt.Tooltip("N",title="N. Students")]
                                                            ).add_selection(highlighter)

    curchart_ci = curchart.mark_rule(color="red").encode(x=alt.X("renewed:O"),y=alt.Y('lbound95'),y2=alt.Y2('ubound95'))
    return curchart_bar + curchart_ci, pval,mean1,mean2

def compare_and_chart_proportion(df,varin,charttitle,legtitle,axtitle,highlighter=highlighter):
    mean1 = float(df.loc[0,(varin,"mean")])
    mean2 = float(df.loc[1,(varin,"mean")])
    phat = (mean1*df.loc[0,("N","count")] + mean2*df.loc[1,("N","count")])/(df.loc[0,("N","count")] + df.loc[1,("N","count")])

    pval = norm.cdf(
                            1-abs(mean1 - mean2)
                                    /np.sqrt(phat*(1-phat)*(1/df.loc[0,("N","count")] + 1/df.loc[1,("N","count")])
                                            )
                            )
    
    df_tochart = df[[('renewed',''),('N','count'),(varin,'mean')]].droplevel(1,axis=1)


    df_tochart["cur_std"] = np.sqrt(
                                        df[varin]["mean"]*(1-df[varin]["mean"])
                                        )/np.sqrt(df["N"]["count"])

    df_tochart["lbound95"] = df[varin]["mean"] \
                                            - 1.961*df_tochart["cur_std"]

    df_tochart["ubound95"] = df[varin]["mean"] \
                                            + 1.961*df_tochart["cur_std"]


    curchart = alt.Chart(df_tochart,width=250,height=250,title=[charttitle,f"p-value: {pval:.3f}","95% C.I. in red"])
    curchart_bar = curchart.mark_bar().encode(x=alt.X("renewed:O"),y=alt.Y(varin + ":Q",scale=alt.Scale(),axis=alt.Axis(title=axtitle)),
                                                                    opacity=alt.condition(~highlighter,alt.value(0.9),alt.value(0.5)),
                                                                    tooltip=[alt.Tooltip(varin,title=legtitle,format=".1f"),
                                                                             alt.Tooltip("cur_std",title="Std. Dev.",format=".2%"),
                                                                             alt.Tooltip("N",title="N. Students")]
                                                            ).add_selection(highlighter)

    curchart_ci = curchart.mark_rule(color="red").encode(x=alt.X("renewed:O"),y=alt.Y('lbound95'),y2=alt.Y2('ubound95'))
    return curchart_bar + curchart_ci, pval,mean1,mean2


########## This is the big SQL query that pulls the basic set of data from the server.
query = """
SELECT 
    tr_summary.userID as userID,                        -- user ID
    tr_summary.startDate as startDate,                  -- user sign-up date
    YEAR(tr_summary.startDate) as startYear,            -- user sign-up year
    level,                                              -- level (2, 3, or 4)
    frac_completed,                                     -- fraction of attempted lessons which are completed
    frac_intervened,                                    -- fraction of attempted lessons which get an intervention
    frac_incomplete,                                    -- fraction of attempted lessons which are left incomplete
    trid_count as lesson_attempts,                      -- number of lesson attempts
    distinct_lesson_count as distinct_lesson_attempts,  -- number of (distinct) lessons attempted
    total_minutes,                                      -- total minutes on program
    frac_trainer_time,                                  -- fraction of time in trainer
    frac_library_time,                                  -- fraction of time in library
    frac_theater_time,
    total_xp,                                           -- etc.
    xp_per_minute,
    total_stars,
    stars_per_minute,
    total_prob_attempts,
    prob_attempts_per_minute,
    frac_correct,
    daid_count as session_count,
    CASE
        WHEN YEAR(endDate) - YEAR(tr_summary.startDate) - (DATE_FORMAT(endDate, '%m%d') < DATE_FORMAT(tr_summary.startDate, '%m%d')) <= 1 THEN 0
        ELSE 1
        END as renewed,
    endDate
FROM
    (
    SELECT
        sd.userID as userID,
        sd.startDate as startDate,
        sd.level as level,
        sd.endDate as endDate,
        sd.subscriptionType as subscr_type,
        AVG(CAST(CASE
                WHEN tr.result = "completed" THEN 1.0
                ELSE 0.0
                END AS FLOAT)) as frac_completed,
        AVG(CAST(CASE
                WHEN tr.result = "intervened" THEN 1.0
                ELSE 0.0
                END AS FLOAT)) as frac_intervened,
        AVG(CAST(CASE
                WHEN tr.result != "completed" AND tr.result != "intervened" THEN 1.0
                ELSE 0.0
                END AS FLOAT)) as frac_incomplete,
        COUNT(tr.id) as trid_count,
        COUNT(DISTINCT tr.lesson) as distinct_lesson_count
    FROM  
        student_data as sd
    INNER JOIN
        trainer_results as tr
    ON sd.userID = tr.userID 
        AND sd.subscriptionType = 'yearly'
        AND DATEDIFF(tr.playDate,sd.startDate) <= 30
        AND YEAR(CURDATE()) - YEAR(sd.startDate) - (DATE_FORMAT(CURDATE(), '%m%d') < DATE_FORMAT(sd.startDate, '%m%d')) > 0
    GROUP BY
        sd.userID
    ) as tr_summary
INNER JOIN
    (
    SELECT
        sd.userID as userID,
        sd.startDate as startDate,
        CAST(SUM(da.trainerTime + da.libraryTime + da.theaterTime) AS FLOAT)/60.0 as total_minutes,
        CAST(SUM(da.trainerTime) AS FLOAT)/CAST(SUM(da.trainerTime + da.libraryTime + da.theaterTime) AS FLOAT) as frac_trainer_time,
        CAST(SUM(da.libraryTime) AS FLOAT)/CAST(SUM(da.trainerTime + da.libraryTime + da.theaterTime) AS FLOAT) as frac_library_time,
        CAST(SUM(da.theaterTime) AS FLOAT)/CAST(SUM(da.trainerTime + da.libraryTime + da.theaterTime) AS FLOAT) as frac_theater_time,
        SUM(xp) as total_xp,
        60.0*CAST(SUM(xp) AS FLOAT)/CAST(SUM(da.trainerTime + da.libraryTime + da.theaterTime) AS FLOAT) as xp_per_minute,
        SUM(starsEarned) as total_stars,
        60.0*CAST(SUM(starsEarned) AS FLOAT)/CAST(SUM(da.trainerTime + da.libraryTime + da.theaterTime) AS FLOAT) as stars_per_minute,
        SUM(problemsAttempted) as total_prob_attempts,
        60.0*CAST(SUM(problemsAttempted) AS FLOAT)/CAST(SUM(da.trainerTime + da.libraryTime + da.theaterTime) AS FLOAT) as prob_attempts_per_minute,
        CAST(SUM(problemsCorrect)/CAST(SUM(problemsAttempted) AS FLOAT) AS FLOAT) as frac_correct,
        COUNT(DISTINCT da.id) as daid_count
    FROM  
        student_data as sd
    INNER JOIN
        daily_activity as da
    ON sd.userID = da.userID 
        AND sd.subscriptionType = 'yearly'
        AND DATEDIFF(da.playDate,sd.startDate) <= 30
        AND YEAR(CURDATE()) - YEAR(sd.startDate) - (DATE_FORMAT(CURDATE(), '%m%d') < DATE_FORMAT(sd.startDate, '%m%d')) > 0
    GROUP BY
        sd.userID
    ) as da_summary
ON tr_summary.userID = da_summary.userID
"""

colnames = ['userID','startDate','startYear','level',
            'frac_completed', 'frac_intervened', 'frac_incomplete','lesson_attempts','distinct_lesson_attempts',
            'total_minutes','frac_trainer_time','frac_library_time','frac_theater_time',
            'total_xp','xp_per_minute','total_stars','stars_per_minute',
            'total_prob_attempts','prob_attempts_per_minute','frac_correct','session_count',
            'renewed','endDate']
df_yearly_student_stats = load_from_sql(query,(sqlserver,sqlport,sqluser,sqlpwd,'students'),colnames)



df_byyear = df_yearly_student_stats.groupby('startYear').agg(dict(zip(['userID'] + colnames[4:],['count'] + [['mean','std']]*18))).reset_index()
df_byyear.rename(columns={'userID':'N'},inplace=True)


df_bylevel = df_yearly_student_stats[df_yearly_student_stats.startYear < 2021].groupby('level').agg(dict(zip(['userID'] + colnames[4:],['count'] + [['mean','std']]*18))).reset_index()
df_bylevel.rename(columns={'userID':'N'},inplace=True)


########## Introduction
st.write("""
### Matt D's Preliminary Report for Beast Academy

In what follows, I will explore a sample of Beast Academy student data. I will attempt to draw some basic, surface-level insights. I will use these to inform recommendations for action in the short, medium, and long term.

Each section of the report is contained in one of the expandable tabs below. Go ahead and click on one to get started!

""")
df_byrenewed = df_yearly_student_stats[df_yearly_student_stats.startYear < 2021].groupby('renewed').agg(dict(zip(['userID'] + colnames[4:-1],['count'] + [['mean','std']]*17))).reset_index()
df_byrenewed.rename(columns={'userID':'N'},inplace=True)


########## Layout
introduction = st.expander("Introduction")
overview = st.expander("Data overview")
timeuse = st.expander("Renewal + time use")
performance = st.expander("Renewal + performance")
discussion = st.expander("Discussion + further steps")



########## Overview content
overview.write("""
#### Data selection

From the database, we select:
 - Only users who purchased yearly subscriptions at least one year ago.
 - Only activity during the 30 days following subcription.

#### Patterns by year
In Figures 1 and 2 we plot number of users and the fraction of renewers for all the years in our sample. 
 - Sample size for 2021 is about half that of previous years. 
 - Renewal rate rises slightly in 2020.
 - Renewal rate calculated as 100% in 2021. This is probably an error due to delayed updating. Therefore, 2021 is excluded from all the remaining analysis.
""")

overview.write(" ")
overview.write(" ")



byyear_tochart = df_byyear[[('startYear',''),('N','count'),('renewed','mean')]].droplevel(1,axis=1)

year_distribution = alt.Chart(byyear_tochart,width=500,height=175,title="Fig. 1: Sampled Yearly Subscriptions")
year_distribution_line = year_distribution.mark_line().encode(x=alt.X("startYear",axis=alt.Axis(format="d",tickMinStep=1)),y=alt.Y("N"))
year_distribution_dots = year_distribution.mark_point().encode(x=alt.X("startYear",axis=alt.Axis(format="d",tickMinStep=1)),y=alt.Y("N"),
                                                                opacity=alt.condition(~highlighter,alt.value(0.9),alt.value(0.5)),
                                                                tooltip=[alt.Tooltip("startYear",title="Starting year"),
                                                                         alt.Tooltip("N",title="No. of yearly subscriptions")]
                                                                ).add_selection(highlighter)
overview.altair_chart(year_distribution_line + year_distribution_dots)

year_renewed = alt.Chart(byyear_tochart,width=500,height=175,title="Fig. 2: Rate of Renewal By Year")
year_renewed_line = year_renewed.mark_line().encode(x=alt.X("startYear",axis=alt.Axis(format="d",tickMinStep=1)),y=alt.Y("renewed"))
year_renewed_dots = year_renewed.mark_point().encode(x=alt.X("startYear",axis=alt.Axis(format="d",tickMinStep=1)),y=alt.Y("renewed"),
                                                                opacity=alt.condition(~highlighter,alt.value(0.9),alt.value(0.5)),
                                                                tooltip=[alt.Tooltip("startYear",title="Starting year"),
                                                                         alt.Tooltip("renewed",title="Renewal rate",format=".1%")]
                                                                ).add_selection(highlighter)
overview.altair_chart(year_renewed_line + year_renewed_dots)


overview.write("""
#### Patterns by level

As can be seen in Figure 3, the rate of renewal is almost exactly the same across student levels. The other statistics which we will discuss in the following sections also do not vary by level.
""")

overview.write(" ")
overview.write(" ")

df_bylevel_tochart = df_bylevel[[('level',''),('N','count'),('renewed','mean')]].droplevel(1,axis=1)

level_renewal = alt.Chart(df_bylevel_tochart,width=400,height=200,title="Fig. 3: Renewal Rate By Level")
year_distribution_bar = level_renewal.mark_bar().encode(x=alt.X("level:O"),y=alt.Y("renewed:Q",scale=alt.Scale(domain=[0,.5])),
                                                                opacity=alt.condition(~highlighter,alt.value(0.9),alt.value(0.5)),
                                                                tooltip=[alt.Tooltip("level",title="Level"),
                                                                         alt.Tooltip("renewed",title="Renewal rate",format=".1%")]
                                                        ).add_selection(highlighter)
overview.altair_chart(year_distribution_bar)

########## Time use content


minutes_chart,pval_min,mean1_min,mean2_min = compare_and_chart_unbounded(df_byrenewed,'total_minutes',"Fig. 4: Time Investment By Renewal","Av. Minutes Used",'minutes',highlighter=highlighter)

trainert_chart,pval_ttr,mean1_ttr,mean2_ttr = compare_and_chart_proportion(df_byrenewed,'frac_trainer_time',"Fig. 5: Trainer Time","% of time on trainer",'Frac. of total time',highlighter=highlighter)

libraryt_chart,pval_libt,mean1_libt,mean2_libt = compare_and_chart_proportion(df_byrenewed,'frac_library_time',"Fig. 6: Library Time","% of time on library",'Frac. of total time',highlighter=highlighter)

theatert_chart,pval_theat,mean1_theat,mean2_theat = compare_and_chart_proportion(df_byrenewed,'frac_theater_time',"Fig. 7: Theater Time","% of time on theater",'Frac. of total time',highlighter=highlighter)

timeuse.write(f"""
First, let's see whether renewers and non-renewers use their time differently.
 - Figure 4: renewers used product for less than non-renewers. {mean2_min:.1f} minutes versus {mean1_min:.1f}.
 - Figures 5, 6, and 7: renewers spend much more time in the library, and a bit less in the trainer or the theater.
""")


timeuse.write(" ")

timeuse.write(" ")


tucolumns = timeuse.columns(2)





tucolumns[0].altair_chart(minutes_chart)


tucolumns[1].altair_chart(trainert_chart)

tucolumns[0].altair_chart(libraryt_chart)

tucolumns[1].altair_chart(theatert_chart)

timeuse.write(" ")


########## Performance content


performance.write("""
Now, let's have a look at the performance of the two groups.
 - Figure 8: renewers attempt problems at slightly faster rate.
 - Figure 9: renewers have a lower rate of correct answers.
""")


percolumns1 = performance.columns(2)

probattempts_chart,pval_probat,mean1_probat,mean2_probat = compare_and_chart_unbounded(df_byrenewed,'prob_attempts_per_minute',"Fig. 8: Problems Attempted Per Min.","Av. Attempts/Min",'rate per minute',highlighter=highlighter)


fraccorrect_chart,pval_corfrac,mean1_corfrac,mean2_corfrac = compare_and_chart_proportion(df_byrenewed,'frac_correct',"Fig. 9: Frac. of Problems Answered Correctly","% of correct answers",'Frac. of total attempts',highlighter=highlighter)


percolumns1[0].altair_chart(probattempts_chart)

percolumns1[1].altair_chart(fraccorrect_chart)

performance.write("""
 - Figures 10 and 11: renewers earn more XP per minute, but fewer stars.
 - Consistent with completing problems faster, but with less success.
""")

percolumns2 = performance.columns(2)

xppermin_chart,pval_xp,mean1_xp,mean2_xp = compare_and_chart_unbounded(df_byrenewed,'xp_per_minute',"Fig. 10: XP Earned Per Minute","Av. XP/Minute",'XP/minute',highlighter=highlighter)


starspermin_chart,pval_stars,mean1_stars,mean2_stars = compare_and_chart_unbounded(df_byrenewed,'stars_per_minute',"Fig. 11: Stars Earned Per Minute","Av. Stars/Minute",'stars/minute',highlighter=highlighter)


percolumns2[0].altair_chart(xppermin_chart)

percolumns2[1].altair_chart(starspermin_chart)




totalsessions_chart,pval_xp,mean1_xp,mean2_xp = compare_and_chart_unbounded(df_byrenewed,'session_count',"Fig. 12: Total Use Sessions","Av. Session Count",'count',highlighter=highlighter)


lessonattempts_chart,pval_stars,mean1_stars,mean2_stars = compare_and_chart_unbounded(df_byrenewed,'lesson_attempts',"Fig. 13: Lessons Attempted","Av. Lessons Tried",'lesson attempts',highlighter=highlighter)



performance.write("""
 - Figure 12: non-renewers do a small but significant additional number of sessions.
 - Figure 13: renewers attempt many fewer lessons
 - This is in line with previous indications of less time spent overall, and in trainer especially.
""")

percolumns3 = performance.columns(2)


percolumns3[0].altair_chart(totalsessions_chart)


percolumns3[1].altair_chart(lessonattempts_chart)




fracintervene_chart,pval_corfrac,mean1_corfrac,mean2_corfrac = compare_and_chart_proportion(df_byrenewed,'frac_intervened',"Fig. 14: Lessons w/ 'Intervention'","% of lessons with intervention",'Frac. of total attempts',highlighter=highlighter)


fracincomplete_chart,pval_corfrac,mean1_corfrac,mean2_corfrac = compare_and_chart_proportion(df_byrenewed,'frac_incomplete',"Fig. 15: Incomplete Lessons","% of lessons left incomplete",'Frac. of total attempts',highlighter=highlighter)



performance.write("""
 - Figures 14 and 15 show no significant differences in the frequency of interventions and abandoned lessons.
 - Point estimate for interventions is a bit higher for renewers, but not significant at 90% or 95% level.
""")

percolumns4 = performance.columns(2)


percolumns4[0].altair_chart(fracintervene_chart)


percolumns4[1].altair_chart(fracincomplete_chart)


########## Discussion content

discussion.write("""
#### Focused strivers versus Experience-builders

**Hypothesis:** renewers motivated by a deeper, multi-purposeful connection with the program.

*Cancellers,* on average:
 - Appear to start with slightly higher aptitude.
 - Move through lessons quickly.
 - Focus on the "trainer," to move forward more efficiently.
 - Want to finish the lessons off, and do not renew once they do so.

*Renewers,* on average:
 - Take a more measure pace.
 - Spend more time in enrichment parts of the program e.g. Library and Theater.

#### Next steps

Immediate recommendations:
 - Why not nudge students towards the library? Low-cost, *might* help.
""")

discussion.write(" ")

discussion.write("""
Short-to-medium term:
 - Use k-means clustering to check for sub-groups of renewers and cancellers.
 - Dig deeper into time series of event reports data, to identify events or patterns which may trigger some cancellations.

""")

discussion.write(" ")

discussion.write("""
Medium-to-long term
 - Train a Logit model on general student characteristics and behavior to identify those who might cancel but could be persuaded.
 - Train another model to look for potentially triggering patterns and events in real time, to help tech and customer support intervene more quickly and efficiently.


""")
