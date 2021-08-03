import json
import numpy as np                        # Used only to find the max and the mean of a matrix (coud be easily done even without it)
import matplotlib.pyplot as plt
import copy

##### CHANGE THIS TO SEE HOW IT WORKS ON OTHER INPUT #####
file = 'input_0.json'

# Opening the input file and (if present) the output file
point_dict = json.load(open(file))
result_file = file.replace('input', 'result')
try:
    result_dict = json.load(open(result_file))
except(FileNotFoundError):
    result_dict = {}

# Division of X and y coordinate of the points
X = np.array(point_dict['X'])
y = np.array(point_dict['Y'])
n_samples = len(X)

# Initialization of the accumulator array
AA = np.zeros((200, 400))
# Instead of using numpy i could have declared a 200*400 list of zeros in a for loop



def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

m = [i for i in range(0, 400, 2)]
# This traslation is to obtain 200 values of m in a range between -2 and 2 since the angle varies from -60 and 60
# because range function takes only integers
m1 = [translate(j, 0, 400, -2, 2) for j in m]


lmerge = tuple(zip(point_dict['X'], point_dict['Y']))

##### UNCOMMENT THIS AND BELOW TO SEE THE PLOT #####
plt.figure('Hough space')
plt.title('Hough space')
plt.xlabel('Angular coefficient m')
plt.ylabel('Intercept q')
plt.grid()

for i in lmerge:
    q = [i[1] - a * i[0] for a in m1]
    for j in range(len(m)):
        if 0 < q[j] < 199 and 0 < m[j] < 399:
            AA[round(q[j]), round(m[j])] += 1
    ##### THIS ARE THE LINES TO UNCOMMENT #####
    plt.plot(m1, q)
plt.show()

# My mesure of the level of noise
noise =100 - (AA.max()-AA.mean())
print('noise = ', noise)

# As treshold i pick the 60% of the max
treshold = AA.max() - AA.max()*40/100

index1=[]
index2 = []
c=0
for n in range(40):
    # If the max is above the trshold i add the index of the matrix corresponding to this max
    if AA[5*n:5*n+5,:].max() >= treshold:
        index1.append(0)
        index2.append(0)
        index1[c],index2[c] = np.unravel_index(AA[5*n:5*n+5,:].argmax(), AA[5*n:5*n+5,:].shape) # Equal to find the index of the max of a list with a loop
        index1[c] += n*5
        c += 1
m_coeff = copy.deepcopy(index2)
q_coeff = copy.deepcopy(index1)
j=0
canceled = False

# Delete the lines which have similar q
for i in range(0, len(index1)-1):
    if canceled:
        canceled = False
        continue
    if index1[i+1]-index1[i] < 6:

    # If i find 2 lines with similar q i delete the one with lower value in the AA
        if AA[index1[i+1],index2[i+1]] > AA[index1[i],index2[i]]:
            del(q_coeff[j])
            del(m_coeff[j])
        else:
            del(q_coeff[j+1])
            del(m_coeff[j+1])
            canceled = True
            j += 1
    else:
        j += 1

# I keep only the lines which have an angular coefficient near to the mean of the angular coefficient of all the lines
m1_index = [translate(j, 0, 400, -2, 2) for j in m_coeff if np.array(m_coeff).mean()-20 < j < np.array(m_coeff).mean()+20]

# The final first initial guess for the angular coefficient is the mean of the angular coefficients of the 'good' lines
m_definitive = np.mean(m1_index)
angle = np.arctan(m_definitive)
angle_deg = angle * 360 / (2 * np.pi)
lines = list(zip(q_coeff,m1_index))

##### UNCOMMENT THIS TO SEE THE PLOTS #####
plt.figure('Initial guess')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Initial guess')
plt.grid()
plt.scatter(point_dict['X'], point_dict['Y'])
for i in range(len(m1_index)):
    plt.plot(X,m_definitive*X+q_coeff[i])
plt.show()
plt.figure('heatmap')
plt.title('Heatmap of AA')
plt.imshow(AA, cmap='hot', interpolation='nearest')

distance_matrix = []
lmerge=list(lmerge)
h = 0
neighbour_points = []
new_q = []
new_m = m_definitive
final_result = []
final_q = []
# print(lmerge)
for i,l in enumerate(lines):
    new_q.append([])
    distance_matrix.append([])
    neighbour_points.append([])
    final_result.append([])
    new_q[i] = l[0]
    lines[i] = list(lines[i])
    lines[i][1] = m_definitive
    final_q.append([])

nearest_line = 0

# Find the neighbour point of the lines
for j,p in enumerate(lmerge):
    min_dist = 1000000
    for i,l in enumerate(lines):
        distance = abs(p[1] - (l[1] * p[0]) - l[0]) / pow(1 + l[1] ** 2, 0.5)
        if distance < min_dist:
            min_dist = distance
            nearest_line = i
    distance_matrix[nearest_line].append(min_dist)
    neighbour_points[nearest_line].append(p)

# Calculation of sum of square distances and standard deviation of the points with respect to the lines of which they are neighbour
std_dev = []
ssd = []
for i,distances in enumerate(distance_matrix):
    ssd.append(sum([dist**2 for dist in distances]))
    std_dev.append(pow(ssd[i],0.5))

# Final refinement for every line
n_iteration = 10000

# Since with the learning rate of q i adjust each line independently from the others i can use an higher learning rate
lrq = 0.1

# Changing the angular coefficient influence all the lines so i have to be more conservative
lrm = 0.001
accumulator_cost = []
accumulator_dcost = []
accumulator_dcost_dm = []
tot_cost = []
d_cost_d_m = [0]
for it in range(n_iteration):
    counter = 0
    intermediate_cost = 0
    update_m = lrm * (sum(d_cost_d_m)/len(d_cost_d_m))

    # I refine the angular coefficient of all the lines toghether
    new_m = new_m - update_m
    d_cost_d_m = []

    for i,l in enumerate(lines):
        cost = 0
        d_cost_d_q = 0
        accumulator_cost = []
        accumulator_dcost = []
        accumulator_dcost_dm = []

        for j,p in enumerate(neighbour_points[i]):
            accumulator_cost.append(pow((-p[1] + (new_m * p[0]) + new_q[i]),2) / (1 + new_m ** 2))
            accumulator_dcost.append(2*(-p[1] + (new_m * p[0]) + new_q[i]) / (1 + new_m ** 2))
            accumulator_dcost_dm.append((2 * (-p[1] + (new_m * p[0]) + new_q[i]) * p[0]) - accumulator_cost[j]*2*new_m)

        cost = sum(accumulator_cost)/(2*len(neighbour_points[i]))

        # The total cost is the sum of the cost of every line
        intermediate_cost += cost
        # The cost of every line (for visualization purpose
        final_result[i].append(cost)
        d_cost_d_q = sum(accumulator_dcost)/(2 * len(neighbour_points[i]))
        d_cost_d_m.append(sum(accumulator_dcost_dm) / (2 * len(neighbour_points[i])))

        # I refine the intercept of every line independentrly
        new_q[i] = new_q[i]-lrq*d_cost_d_q
        final_q[i] = new_q[i]

        # If i'm not updating the q for all the lines i will see if i'm not updating also m
        if abs(lrq*d_cost_d_q) < 0.0001:
            counter+=1
    # if it % 100 == 0:
    #     print(cost)

    # Total cost for visualization purpose
    tot_cost.append(intermediate_cost)

    # If i'm not updating also m i can break the cycle
    if counter == len(lines) and abs(update_m < 0.0001):
        break


angle = np.arctan(new_m)
angle_deg = angle * 360 / (2 * np.pi)

print('Here there are the obtained result: ')
print('\tThe lines I found have an angle with the x axis of: ',angle_deg)
if len(result_dict)!=0:
    print('\tThe ideal lines have an angle with the x axis of: ', result_dict['angle'])
    print('\tThe error in my estimation of the angle is: ', abs(angle_deg - result_dict['angle']))
print('\tThe intercept of the lines I found are: ', final_q)
if len(result_dict)!=0:
    print('\tThe ideal intercept of the lines are: ', result_dict['y_0'])
    print('\tThe error in my estimation of the intercept is: ', [final_q[i]-result_dict['y_0'][i] for i in range(len(final_q))])
else:
    print('No data available for comparison')

##### UNCOMMENT THIS TO SEE THE PLOTS #####
plt.figure('figure 2')
plt.title('Optimized result')
plt.xlabel('X')
plt.ylabel('y')
plt.grid()
plt.scatter(point_dict['X'], point_dict['Y'])
for i in range(len(lines)):
    plt.plot(X,new_m*X+final_q[i])
plt.show()
plt.figure('loss')
plt.title('Loss function of each line')
plt.xlabel('Iterations')
plt.ylabel('Costs')
plt.grid()
for i in range(len(final_result)):
    plt.plot(range(len(final_result[i])-1),final_result[i][1:])
plt.show()
plt.figure('tloss')
plt.title('Total loss function')
plt.xlabel('Iterations')
plt.ylabel('Total cost')
plt.grid()
plt.plot(range(len(tot_cost)-1),tot_cost[1:])
plt.show()
