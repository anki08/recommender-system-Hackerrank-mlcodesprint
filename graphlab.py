
import pandas as pd
import graphlab
submissions = pd.read_csv('submissions.csv')
print(submissions.head())
print(submissions.head())
print(submissions.info())
print(submissions.describe())
# consider ques solvedby the hacker
Q_solv = submissions[['hacker_id', 'contest_id', 'challenge_id', 'solved']]
# consider only those challenges having contest_id = c8ff662c97d345d2
Q_solv = Q_solv[Q_solv['contest_id'] == 'c8ff662c97d345d2']
# drop contest_id as all have same contest_id
Q_solv.drop('contest_id', axis=1, inplace=True)
print(Q_solv.head(20))
Q_solv.sort_values(['hacker_id', 'challenge_id'], ascending=True)
Q_solv.to_csv('yo1.csv', index=False)
# we see that the challenge_id has repeated values and 'solved' has different
# values for it indicating that the hacker_id solved the question later
# so we change the values of solved and delete the duplicate values
Q_solv['Solved'] = Q_solv.groupby(['hacker_id', 'challenge_id'])[
    'solved'].transform('max')
Q_solv.drop('solved', axis=1)
Q_solv = Q_solv.drop_duplicates()
# we are intersted in only those rows where the hacker didnot solve completely
Q_solv = Q_solv[Q_solv['Solved'] == 0]
# we add created at(time) to the dataframe
time = submissions[['hacker_id', 'challenge_id', 'created_at']]
Q_solv = pd.merge(Q_solv, time, on=['hacker_id', 'challenge_id'])
# We consider the last q attempted as the person is trying to solve that group
# of problems. Now we have the max time of each challenge_id
Q_solv['time'] = Q_solv.groupby(['hacker_id', 'challenge_id'])[
    'created_at'].transform('max')
# as we considered only those ques we hadnt solved we drop Score
Q_solv = Q_solv[['hacker_id', 'challenge_id', 'time']]
# drop duplicates and reorder index
Q_solv = Q_solv.drop_duplicates().reset_index(drop=True)
# we find the last type of problem solved byt he hacker if that type finishes,
# we go to the 1 he was solving before.So we sort the dataframe in
# descending order
Q_solv = Q_solv.sort_values(['hacker_id', 'time'], ascending=[False, False])
print(Q_solv.info())
print(Q_solv.head())
# list of hackers having unsolved ques
unsolved_hacker_list = list(Q_solv['hacker_id'].unique())

# loading challenges.csv
challenges = pd.read_csv('challenges.csv')
# added feature ratio of solved_submissions_count/solved-submissions_count
challenges['ratio'] = challenges['solved_submission_count'] / \
    challenges['total_submissions_count']
challenges['ratio'] = challenges['ratio'].astype(float)
df = pd.merge(submissions, challenges, on=[
              'contest_id', 'challenge_id'], how='left')
print(df.info())
# drop domain and subdomain as a lot of values are missing
df.drop(['total_submissions_count', 'solved_submission_count',
         'domain', 'subdomain'], axis=1, inplace=True)
print(df.head())
# train consists of all the data where solved = 1
train_data = graphlab.SFrame(df[df['solved'] == 1])
model = graphlab.item_similarity_recommender.create(train_data,
                                                    user_id='hacker_id',
                                                    item_id='challenge_id',
                                                    similarity_type='jaccard')
                                                    
score = model.evaluate(train_data)
print(score)

users_list = list(submissions['hacker_id'].unique())
recommendation = model.recommend(users=users_list, k=50)
score = model.evaluate(train_data)
print(score)
# ensure that all predicted values are from the given list of challenges
challenge_contest = challenges[['challenge_id', 'contest_id']]
challenge_contest = challenge_contest.drop_duplicates()
recommendation_df = recommendation.to_dataframe()
recommendation_df = pd.merge(
    recommendation_df, challenge_contest, on='challenge_id', how='left')
# keep only valid contest_id
recommendation_df = recommendation_df[
    recommendation_df['contest_id'] == 'c8ff662c97d345d2']
recommendation_df.drop(['contest_id', 'rank'], axis=1, inplace=True)
# find out no of challenges recommended for each user
recommendation_df['hacker_count'] = recommendation_df.groupby(
    ['hacker_id'])['challenge_id'].transform('count')
print(recommendation_df['hacker_count'].describe())
# recommendation_df.to_csv('yo111.csv',index=False)
print(len(users_list))
# the no of recommendations vary from 2 to 50
recommendation_df['challenge_count'] = recommendation_df.groupby(
    ['challenge_id'])['score'].transform('count')
challenge_counts = set(list(recommendation_df['challenge_count']))
# hackers with less than 10 recommendations
less_recommends = recommendation_df[recommendation_df['hacker_count'] < 10]
less_hacker_list = list(less_recommends['hacker_id'].unique())
print(len(less_hacker_list))
pred_hacker_list = list(recommendation_df['hacker_id'].unique())
print(len(pred_hacker_list))
total_hacker_list = list(submissions['hacker_id'].unique())
# there are 2 hackers with no recommendation
missing_hacker_list = [
    i for i in total_hacker_list if i not in pred_hacker_list]
print(missing_hacker_list)
# hackers for whom 10 recommendations are available
ten_recomm_hacker_list = [
    i for i in pred_hacker_list if i not in less_hacker_list]
print(len(ten_recomm_hacker_list))
# results for hackers with 10 recommendations
# 2 unsolved challenge is added , saw this feature in interviewbit.com
result = []
for user in ten_recomm_hacker_list:
    temp = recommendation_df[recommendation_df[
        'hacker_id'] == user].reset_index(drop=True)
    if user in unsolved_hacker_list:
        unsolved_list = list(Q_solv[Q_solv['hacker_id'] == user][
                             'challenge_id'].unique())
        a = len(unsolved_list)
        b = 0
        if a > 2:
            unsolved_list = unsolved_list[:2]
        a = len(unsolved_list)
        res = [user]
        res = res + unsolved_list
        while a < 10:
            c = temp['challenge_id'][b]
            if c not in unsolved_list:
                res.append(c)
                a += 1
            b += 1
    else:
        res = [user]
        i = 0
        for i in range(10):
            res.append(temp['challenge_id'][i])
    reso = res[0] + ',' + res[1] + ',' + res[2] + ',' + res[3] + ',' + res[4] + ',' + \
        res[5] + ',' + res[6] + ',' + res[7] + ',' + \
        res[8] + ',' + res[9] + ',' + res[10]
    result.append(reso)

# for users with less recommendations unsolved challenges are added
most_solved_challenges = []
for i in (sorted(challenge_counts)[-10:]):
    j = recommendation_df[recommendation_df['challenge_count'] == i].head(1)[
        'challenge_id'].values[0]
    most_solved_challenges.append(j)

for user in less_hacker_list:
    temp_df = recommendation_df[recommendation_df[
        'hacker_id'] == user].reset_index(drop=True)
    res = [user]
    res = res + list(temp_df['challenge_id'])
    a = (len(res) - 1)
    # if no of recommended challenges are also less than 10
    while a < 10:
        res.append(most_solved_challenges[a])
        a += 1
    reso = res[0] + ',' + res[1] + ',' + res[2] + ',' + res[3] + ',' + res[4] + ',' + \
        res[5] + ',' + res[6] + ',' + res[7] + ',' + \
        res[8] + ',' + res[9] + ',' + res[10]
    result.append(reso)

# for hackers with missing no of challenges
for user in missing_hacker_list:
    res = [user] + most_solved_challenges
    reso = res[0] + ',' + res[1] + ',' + res[2] + ',' + res[3] + ',' + res[4] + ',' + \
        res[5] + ',' + res[6] + ',' + res[7] + ',' + \
        res[8] + ',' + res[9] + ',' + res[10]
    result.append(reso)

result = pd.DataFrame(result)
#print(graphlab.evaluation.confusion_matrix(graphlab.SArray(targets), graphlab.SArray(predictions), average='macro'))
print(result.shape)
# print result
result.to_csv('recommendation.csv', header=False, index=False)
