# pip install pandas
import copy
import random
import numpy as np
import pandas as pd
from scipy import stats

# Replace 'your_file.csv' with the path to your CSV file
data = pd.DataFrame(pd.read_csv('flightdelay.csv'))
flight_columns = {}

max_depth =  10
target = 'DEP_DEL15'
alpha = 0.05
debug = False
debug_level=2


def print_(string,level = 0):
    if debug and level<debug_level:
        print(string)

class Range():
    def __init__(self, nane, bin_=[], manipulation_function=lambda x: x):
        self.name = nane  ### str
        self.bins = bin_  ### [ [x1,x2] ,[x3,x4,x5] ]  or continuous [ [[x1,x2]] ,[[x3,x4],[x7,x8]] , [[x5,x6]] ]
        self.manipulation_function = manipulation_function

    def data_manipulation(self, coll):
        return coll.apply(lambda x: self.manipulation_function(x))

class Q1():
    def __init__(self):
        self.flight_columns = []
        self.df = pd.DataFrame(pd.read_csv('flightdelay.csv'))
        print_('upload data')
        self.range_data()
        self.prepared_data = self.prepare_data(self.df)
        self.tree = None

    def range_data(self):

        # _____________MONTH______________
        self.flight_columns.append(Range('MONTH', [[6, 2, 7, 12, 8, 5, 4],[ 1, 3, 10, 11, 9]]))
        # _____________DAY_OF_WEEK______________
        self.flight_columns.append(Range('DAY_OF_WEEK', [[4, 5, 1, 7], [3, 2, 6]]))
        # _____________DEP_TIME_BLK______________
        # self.flight_columns.append(Range('DEP_TIME_BLK', [['1900-1959', '2100-2159', '2000-2059', '1800-1859', '1700-1759', '1600-1659', '2200-2259'],[ '1500-1559', '1400-1459', '2300-2359', '1300-1359', '1200-1259', '1100-1159', '1000-1059'],[ '0900-0959', '0800-0859', '0700-0759', '0001-0559', '0600-0659']]))
        # _____________DISTANCE_GROUP______________
        self.flight_columns.append(Range('DISTANCE_GROUP', [[7, 5, 8, 9, 4, 6, 3], [11, 10, 2, 1]]))
        # _____________SEGMENT_NUMBER______________
        self.flight_columns.append(Range('SEGMENT_NUMBER', [[7],[ 5, 9, 6, 8, 15, 4],[ 3, 2, 10, 1],[ 12, 11, 13, 14]]))
        # _____________CONCURRENT_FLIGHTS______________
        self.flight_columns.append(Range('CONCURRENT_FLIGHTS', [[[1, 30]], [[30, 109]]]))
        # _____________NUMBER_OF_SEATS______________
        self.flight_columns.append(Range('NUMBER_OF_SEATS', [[[44, 100]], [[100, 337]]]))
        # _____________CARRIER_NAME______________
        self.flight_columns.append(Range('CARRIER_NAME',
                                         [['Frontier Airlines Inc.', 'JetBlue Airways', 'Atlantic Southeast Airlines'],
                                          ['Southwest Airlines Co.', 'American Airlines Inc.', 'Comair Inc.',
                                           'United Air Lines Inc.', 'Spirit Air Lines', 'Mesa Airlines Inc.',
                                           'Allegiant Air', 'American Eagle Airlines Inc.'],
                                          ['SkyWest Airlines Inc.', 'Endeavor Air Inc.', 'Alaska Airlines Inc.',
                                           'Midwest Airline, Inc.', 'Delta Air Lines Inc.', 'Hawaiian Airlines Inc.']]))
        # _____________AIRPORT_FLIGHTS_MONTH______________
        self.flight_columns.append(
            Range('AIRPORT_FLIGHTS_MONTH', [[[1100, 25000]], [[25000, 35256]]]))
        # _____________AIRLINE_FLIGHTS_MONTH______________
        self.flight_columns.append(Range('AIRLINE_FLIGHTS_MONTH', [[[5582,70000]],[[70000,117728]]]))
        # _____________AIRLINE_AIRPORT_FLIGHTS_MONTH______________
        self.flight_columns.append(Range('AIRLINE_AIRPORT_FLIGHTS_MONTH', [[[1,15000]],[[15000,21837]]]))
        # _____________AVG_MONTHLY_PASS_AIRPORT______________
        self.flight_columns.append(Range('AVG_MONTHLY_PASS_AIRPORT', [[[70476,3365661]],[[3365661,4365661]]]))
        # _____________AVG_MONTHLY_PASS_AIRLINE______________
        self.flight_columns.append(Range('AVG_MONTHLY_PASS_AIRLINE',
                                         [[1857122, 3190369, 473794, 13382999, 11744595, 1245396,8501631, 2688839, 1191889],
                                          [ 1257616, 1204766, 3472966,1212846, 2884187, 1529740, 12460183, 905990] ]))

        # _____________FLT_ATTENDANTS_PER_PASS______________
        self.flight_columns.append(Range('FLT_ATTENDANTS_PER_PASS', [[0.0001157256482767, 0.0001600389254787],
                                                                     [6.178236301460919e-05, 9.82082928995461e-05,
                                                                      0.0002538042406215, 9.173723925704242e-06, 0.0,
                                                                      1.3252580061983644e-05, 0.000348407665605,
                                                                      3.419267401443636e-05],
                                                                     [3.233146156611231e-05, 1.25293581928874e-06,
                                                                      0.0001441658849878], [0.0001204942176112]]))
        # _____________GROUND_SERV_PER_PASS______________
        self.flight_columns.append(Range('GROUND_SERV_PER_PASS', [
            [7.134694872433899e-06, 0.0001268661761139, 0.0001998053656534, 9.88941230999822e-05],
            [0.000177287219593, 8.999810569042966e-05, 0.0002289854734932, 0.0001246510730715, 9.131157116170832e-05,
             0.0001077434759039, 0.000106867151749, 9.900278805864172e-05, 9.351277617445625e-05, 0.0001746014497265,
             0.0001238227442279, 0.0001486602009422, 0.0001978496657113]]))
        # _____________PLANE_AGE______________
        self.flight_columns.append(Range('PLANE_AGE', [[[0, 10]], [[10, 32]]]))
        # _____________DEPARTING_AIRPORT______________
        self.flight_columns.append(Range('DEPARTING_AIRPORT', [
            ['Chicago Midway International', 'Myrtle Beach International', 'Newark Liberty International',
             'William P Hobby', 'Dallas Love Field', 'Fort Lauderdale-Hollywood International', 'LaGuardia',
             "Chicago O'Hare International", 'Dallas Fort Worth Regional', 'Stapleton International',
             'San Francisco International', 'Lambert-St. Louis International', 'Logan International',
             'Metropolitan Oakland International', 'Friendship International', 'Orlando International',
             'Portland International Jetport', 'Palm Beach International', 'Pensacola Regional',
             'Louis Armstrong New Orleans International', 'McCarran International', 'Douglas Municipal',
             'Nashville International', 'Houston Intercontinental', 'Los Angeles International',
             'Phoenix Sky Harbor International', 'John F. Kennedy International', 'Raleigh-Durham International',
             'Rochester Monroe County', 'Greenville-Spartanburg', 'Southwest Florida International',
             'Miami International'],
            ['Tampa International', 'Austin - Bergstrom International', 'Ronald Reagan Washington National',
             'Cincinnati/Northern Kentucky International', 'Palm Springs International', 'Richmond International',
             'Savannah/Hilton Head International', 'Philadelphia International', 'Theodore Francis Green State',
             'Northwest Arkansas Regional', 'San Diego International Lindbergh Fl', 'Birmingham Airport',
             'Albany International', 'Norfolk International', 'Hollywood-Burbank Midpoint', 'Seattle International',
             'James M Cox/Dayton International', 'Cleveland-Hopkins International', 'Atlanta Municipal',
             'Memphis International', 'McGhee Tyson', 'Will Rogers World', 'El Paso International', 'Kent County',
             'Pittsburgh International', 'General Mitchell Field', 'Greater Buffalo International',
             'Port Columbus International', 'Piedmont Triad International', 'Indianapolis Muni/Weir Cook',
             'Charleston International',
             'Albuquerque International Sunport', 'Detroit Metro Wayne County', 'Standiford Field', 'Orange County',
             'Bradley International', 'San Antonio International', 'Ontario International',
             'Minneapolis-St Paul International', 'Eppley Airfield', 'Jacksonville International',
             'Des Moines Municipal', 'Kansas City International', 'Adams Field', 'Washington Dulles International',
             'Syracuse Hancock International', 'Truax Field', 'Sacramento International', 'San Jose International',
             'Reno/Tahoe International', 'Tulsa International', 'Salt Lake City International',
             'Long Beach Daugherty Field', 'Tucson International', 'Portland International', 'Boise Air Terminal',
             'Spokane International', 'Sanford NAS', 'Kahului Airport', 'Honolulu International',
             'Puerto Rico International', 'Lihue Airport', 'Anchorage International', 'Keahole']]))
        # _____________LATITUDE______________
        self.flight_columns.append(Range('LATITUDE', [[[0, 25]], [[25, 61.169]]]))
        # _____________LONGITUDE______________
        self.flight_columns.append(Range('LONGITUDE', [[[-159.346, -100]], [[-100, -66.002]]]))
        # _____________PREVIOUS_AIRPORT______________
        self.flight_columns.append(Range('PREVIOUS_AIRPORT', [
            ['Concord Regional', 'Barnstable Municipal-Boardman/Polando Field', 'Salina Municipal'],
            ['Vernal Regional', 'Lockbourne AFB', "Martha's Vineyard Airport", 'Devils Lake Regional',
             'Gillette Campbell County', 'Houghton County Memorial', 'Worcester Regional', 'Plattsburgh International',
             'LaGuardia', 'Newark Liberty International', 'Westchester County', 'Waterloo Regional',
             'Aspen Pitkin County Sardy Field', 'Yampa Valley', 'Hays Municipal', 'Branson Airport',
             'Puerto Rico International', 'Nantucket Memorial', 'Benedum', 'Chicago Midway International',
             'Fort Lauderdale-Hollywood International', 'Liberal Municipal', 'Palm Beach International', 'Arcata',
             'Dallas Love Field', 'Orlando International', 'Logan International', 'Montrose Regional',
             'San Francisco International', 'Eagle County Regional', 'William P Hobby', 'Mercer County',
             "Chicago O'Hare International", 'Cheyenne Regional/Jerry Olson Field', 'Key Field',
             'Dallas Fort Worth Regional', 'Tri-State/Milton J. Ferguson Field', 'Southwest Georgia Regional',
             'North Bend Municipal', 'Quincy Regional-Baldwin Field', 'Louis Armstrong New Orleans International',
             'Albany International', 'Barkley Regional', 'Jackson Hole', 'Eau Claire Municipal', 'Alexander Hamilton',
             'Roanoke Regional/Woodrum Field', 'Friendship International', 'Pierre Municipal',
             'Stapleton International', 'Philadelphia International', 'Albuquerque International Sunport',
             'Bangor International', 'Lambert-St. Louis International', 'Stewart International',
             'John F. Kennedy International', 'Nashville International', 'Peterson Field',
             'Southwest Florida International', 'Miami International', 'Harry S Truman', 'San Antonio International',
             'Hollywood-Burbank Midpoint', 'Lincoln Airport', 'McCarran International',
             'San Diego International Lindbergh Fl', 'Midland Regional Air Trml', 'Ronald Reagan Washington National',
             'Raleigh-Durham International', 'Tampa International', 'Long Island MacArthur', 'Valley International',
             'Austin - Bergstrom International', 'Mobile Aerospace', 'Cherry Capital', 'Orange County',
             'Metropolitan Oakland International', 'Cedar Rapids Municipal', 'Will Rogers World',
             'El Paso International', 'Kansas City International', 'Houston Intercontinental', 'Amarillo International',
             'General Mitchell Field', 'Allen C Thompson Field', 'Wilkes Barre Scranton International',
             'Valdosta Regional', 'Kanawha', 'Theodore Francis Green State', 'Greater Peoria', 'Sanford NAS',
             'James M Cox/Dayton International', 'Birmingham Airport', 'Greater Buffalo International',
             'Portland International Jetport', 'Columbia Regional', 'Cape Girardeau Regional',
             'Northwest Florida Beaches International', 'Ontario International', 'Savannah/Hilton Head International',
             'Phoenix Sky Harbor International', 'Tulsa International', 'Los Angeles International',
             'Douglas Municipal', 'General Brees Field', 'Medford Jackson County', 'Nafec Atlantic City',
             'Norfolk International', 'Bradley International', 'Eppley Airfield', 'Sarasota/Bradenton Airport',
             'Cincinnati/Northern Kentucky International', 'Pitt Greenville', 'Jacksonville International',
             'Charleston International', 'Cleveland-Hopkins International', 'Quad City International',
             'Memphis International', 'Des Moines Municipal', 'Prescott Municipal', 'Palm Springs International',
             'Kent County', 'Manhattan Regional', 'Springfield-Branson National', 'Standiford Field',
             'Hilton Head Airport', 'Mahlon Sweet Field', 'Akron-Canton Regional', 'McGhee Tyson', 'Latrobe Airport',
             'Richmond International', 'Dubuque Regional', 'Ramey AFB', 'Muskegon County', 'Williams Gateway',
             'Scott AFB', 'Pittsburgh International', 'Seattle International', 'Fort Wayne Municipal',
             'San Luis Obispo County Regional','Minneapolis-St Paul International', 'Asheville Regional', 'Erie International',
             'Long Beach Daugherty Field', 'Sacramento International', 'Myrtle Beach International', 'Kincheloe AFB',
             'Sioux City Gateway', 'McAllen Miller International', 'Northwest Florida Regional', 'Joplin Municipal',
             'Washington Dulles International', 'Evansville Dress Regional', 'Port Columbus International',
             'Elmira/Corning Regional', 'Delta County', 'Adams Field', 'Gulfport-Biloxi International', 'Meadows Field',
             'Indianapolis Muni/Weir Cook', 'Tucson International', 'Pensacola Regional', 'Wichita Mid-Continent',
             'Agana Field', 'Dannelly Field', 'Shreveport Regional', 'North Platte Regional Airport Lee Bird Field',
             'Tri City', 'Charlottesville Albemarle', 'Gallatin Field', 'Melbourne Regional', 'Ponce Airport',
             'Watertown Municipal', 'Portsmouth Pease International', 'Mammoth Lakes Airport',
             'Hattiesburg-Laurel Regional', 'Detroit Metro Wayne County', 'Central Illinois Regional',
             'Greenville-Spartanburg', 'Santa Barbara Municipal', 'Canyonlands Field',
             'Kalamazoo/Battle Creek International', 'Bishop International', 'Phelps/Collins', 'Greenbrier Valley',
             'Burlington International', 'Capital City', 'Lubbock Regional', 'Reno/Tahoe International',
             'Redding Municipal', 'Capital', 'San Jose International', 'Bemidji Municipal', 'Michiana Regional',
             'Duluth International', 'Shenandoah Valley Regional', 'Watertown International',
             'Austin Straubel International', 'Roswell International Air Center', 'Northwest Arkansas Regional',
             'Atlanta Municipal', 'Gunnison-Crested Butte Regional', 'La Crosse Municipal', 'Harrisburg International',
             'Spokane International', 'Jamestown Regional', 'Pellston Regional Airport of Emmet County',
             'Monterey Peninsula', 'Rapid City Regional', 'Wilmington International', 'Salt Lake City International',
             'Glacier Park International', 'Rochester Monroe County', 'Joe Foss Field', 'Lafayette Regional',
             'Portland International', 'Marquette County', 'Corpus Christi International',
             'Syracuse Hancock International', 'Missoula International', 'Ford', 'Grenier Field/Manchester Municipal',
             'Bismarck Municipal', 'Mobile Regional', 'Truax Field', 'Billings Logan International',
             'Aberdeen Regional', 'Rock Springs Sweetwater County', 'La Plata County', 'Abilene Regional',
             'Golden Triangle Regional', 'Columbia Metropolitan', 'Huntsville International-Carl T Jones Field',
             'State College Air Depot', 'Lovell Field', 'Monroe Regional', 'Piedmont Triad International',
             'Lehigh Valley International', 'Fayetteville Regional/Grannis Field', 'Anchorage International',
             'Glynco Jetport', 'Lea County Hobbs', 'Tallahassee Regional', 'St. Petersburg-Clearwater International',
             'Daytona Beach Regional', 'Fresno Air Terminal', 'Fort Smith Regional', 'Rochester Municipal',
             'Snohomish County Paine Field', 'Texarkana Muni/Webb Field', 'William B. Heilig Field',
             'Santa Fe Municipal', 'Baton Rouge Metropolitan/Ryan Field', 'University of Illinois/Willard',
             'Blue Grass', 'Laredo AFB', 'Grand Forks International', 'Roberts Field', 'Hector Field',
             'Greater Binghamton/Edwin A. Link Field', 'Dothan Regional', 'Ithaca Tompkins Regional',
             'Charlotte County', 'Chicago/Rockford International', 'Santa Maria Public/Capt. G. Allan Hancock Field',
             'Tri Cities', 'Craven County Regional', 'Patrick Henry International', 'Easterwood Field',
             'Tyler Pounds Regional', 'Charles M. Schulz - Sonoma County', 'Key West International',
             'Outagamie County Regional', 'Tri-Cities Regional TN/VA', 'Gainesville Regional',
             'Lawton-Fort Sill Regional', 'Ogdensburg International', 'Albert J Ellis', 'Boise Air Terminal',
             'Jefferson County', 'Del Rio International', 'Minot International', 'Walker Field', 'Robert Gray AAF',
             'Bush Field', 'Provo Municipal', 'San Angelo Regional/Mathis Field',
             'Brownsville South Padre Island International', 'Kearney Regional', 'Fanning Field',
             'Lynchburg Regional/Preston Glenn Field', 'Columbus Metropolitan', 'NONE', 'Toledo Express',
             'Tweed New Haven', 'East Texas Regional', 'Ketchikan International', 'Cedar City Regional',
             'Stockton Metropolitan', 'Niagara Falls International', 'Central Wisconsin',
             'Sheppard AFB/Wichita Falls Municipal', 'Waco Regional', 'Juneau International', 'Fairbanks International',
             'St George Municipal', 'Flagstaff Pulliam', 'Friedman Memorial', 'Pueblo Memorial',
             'Sloulin Field International', 'Yuma MCAS/Yuma International', 'Kahului Airport', 'England AFB',
             'Joslin Field - Magic Valley Regional', 'Lake Charles Regional', 'Bellingham International',
             'Falls International', 'Grand Island Air Park', 'Great Falls International', 'Pocatello Regional',
             'Casper/Natrona County International', 'Yellowstone', 'Kodiak Airport', 'Honolulu International',
             'Rhinelander/Oneida County', 'Searcy Field', 'Lihue Airport', 'Sitka Rocky Gutierrez', 'Chisholm/Hibbing',
             'Keahole', 'Cordova Mile 13', 'Elko Regional', 'Bethel Airport', 'Yellowstone Regional',
             'Lewiston Nez Perce County', 'Helena Regional', 'Williston Basin International', 'Deadhorse Airport',
             'Brainerd/Crow Wing County', 'Wiley Post/Will Rogers Memorial', 'Garden City Municipal', 'Nome Airport',
             'General Lyman Field', 'Bert Mooney', 'King Salmon Airport', 'Ralph Wien Memorial',
             "Hagerstown Regional-Richard A. Henson Field", "Pago Pago International", 'Owensboro Daviess County',
             'Adak NS', 'Dillingham Airport']]))
        # _____________PRCP______________
        self.flight_columns.append(Range('PRCP', [[[0.0, 2]], [[2, 11.63]]]))
        # _____________SNOW______________
        self.flight_columns.append(Range('SNOW', [[[0.0, 4]], [[4, 17.2]]]))
        # _____________SNWD______________
        self.flight_columns.append(Range('SNWD', [[[0.0, 5.9]], [[5.9, 20.9]], [[20.9, 252]]]))
        # _____________TMAX______________
        self.flight_columns.append(Range('TMAX', [[[-100.0, 0]], [[0, 1150]]]))
        # _____________AWND______________
        self.flight_columns.append(Range('AWND', [[[0.0, 2.9]],[[2.9,3.1],[22.8,26]], [[3.1, 22.8]]]))
    def prepare_data(self, data):
        prepared_data = pd.DataFrame()
        if target in data:
            prepared_data[target] = data[target]
        for filled in self.flight_columns:
            bins = filled.bins
            if bins:
                if type(bins[0][0]) == list:
                    def find_bin(x):
                        for index, bin_ in enumerate(bins):
                            for range_ in bin_:
                                if range_[0] <= x < range_[1]:
                                    return index

                else:
                    def find_bin(x):
                        for index, bin_ in enumerate(bins):
                            if x in bin_:
                                return index
                prepared_data[filled.name] = data[filled.name].apply(lambda x: find_bin(x))
        return prepared_data

    def get_ratio_of_rows_shuffled(self, df, ratio):
        random.seed(123)
        # Calculate the number of rows to return based on the ratio
        num_rows = int(len(df) * ratio)
        # Shuffle the DataFrame rows
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        # Split the shuffled DataFrame into two parts
        df_ratio = shuffled_df.head(num_rows)
        df_remaining = shuffled_df.tail(len(df) - num_rows)
        return df_ratio, df_remaining


class DecisionTreeNode:
    def __init__(self, feature_index=None, children=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.children = children  # Threshold value to split on
        self.value = value  # Value if it's a leaf node

    def __repr__(self):
        return (f'feature :{self.feature_index} \n children : {self.children}')


class DecisionTree:
    def __init__(self):
        self.tree = {}
        self.max_depth = max_depth

    def build_tree(self, examples, attributes, target, depth=0):
        # Check if examples is empty
        value = int(examples[target].iloc[0])
        if examples.empty:
            return True, None
        # Check if all target values are the same
        elif len(examples[target].unique()) == 1:
            return True, DecisionTreeNode(value=value)
        # If no more features to split, return the most common target value
        elif len(examples.columns) == 1 or depth > self.max_depth:
            return True, DecisionTreeNode(value=examples[target].mode()[0])
        # Select the best attribute (placeholder, implement your method to choose the best attribute)
        best_attribute = self.best_attribute(examples, attributes, target)
        tree = {}
        # Split examples based on the best attribute and build subtree recursively
        for attribute_value, sub_examples in examples.groupby(best_attribute):
            sub_examples = sub_examples.drop(columns=[best_attribute])  # Drop the used attribute
            leave, subtree = self.build_tree(sub_examples, [attribute for attribute in attributes if attribute != best_attribute], target, depth + 1)
            if subtree is not None:
                if leave:
                    avg_values = sub_examples[target].agg(['sum', 'count'])
                    if self.chi_2(examples[target].mean(), avg_values['sum'],
                                  avg_values['count'] - avg_values['sum']):
                        tree[attribute_value] = subtree

                else:
                    tree[attribute_value] = subtree
        if not tree: tree = None
        return tree is None, DecisionTreeNode(best_attribute, tree, value)

    def fit(self, examples, target):
        print_('fitting model')
        attributes = list(examples.keys())
        attributes.remove(target)
        _, self.tree = self.build_tree(examples, attributes, target)
        if self.tree.value is None:
            self.tree.value = examples[target].iloc[0]
        print_(f'{self.tree}',1)
        print_("free found")
        return self.tree

    def compute_weights(self, examples, attributes, target,size,father_entropy):
        weights = []
        for attribute in attributes:
            weight = 0
            avg_values = examples.groupby(attribute)[target].agg(['mean', 'count'])
            for action, row in avg_values.iterrows():
                p = row['mean']
                n = row['count']
                weight += self.calculate_entropy(p) * n / size
            weights.append({'attribute': attribute, 'weight': father_entropy - weight})
        return weights

    def calculate_entropy(self, p):
        if p == 0 or p == 1: return 0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def best_attribute(self, examples, attributes, target):
        size = examples[target].count()
        father_entropy = self.calculate_entropy(examples[target].mean())
        return max(self.compute_weights(examples, attributes, target,size,father_entropy), key=lambda x: x['weight'])['attribute']

    def chi_2(self, Tr, T, F):
        n = T + F
        observed_frequencies = [T, F]
        # Expected frequencies
        expected_frequencies = [Tr * n, (1 - Tr) * n]
        # Perform the chi-squared test
        _, p = stats.chisquare(observed_frequencies, f_exp=expected_frequencies)
        # return True
        return p > alpha

    def _predict(self, inputs, node):
        if node.children is not None:
            value = inputs[node.feature_index]
            if value in node.children.keys():
                return self._predict(inputs, node.children[value])
        return node.value

    def tree_error(self, tree, df_test):
        df = pd.DataFrame()
        df['actuals'] = df_test[target]
        df['predictions'] = [self._predict(inputs, tree) for index, inputs in df_test.iterrows()]
        accuracy, precision, recall, f1 = self.evaluate(df['predictions'], df['actuals'])
        return accuracy, precision, recall, f1

    def evaluate(self, predictions, actuals):
        true_positive = sum((predictions == 1) & (actuals == 1))
        true_negative = sum((predictions == 0) & (actuals == 0))
        false_positive = sum((predictions == 1) & (actuals == 0))
        false_negative = sum((predictions == 0) & (actuals == 1))

        print_(f"true_positive : {true_positive}",1)
        print_(f"true_negative : {true_negative}",1)
        print_(f"false_positive : {false_positive}",1)
        print_(f"false_negative : {false_negative}",1)

        accuracy = (true_positive + true_negative) / len(actuals)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return accuracy, precision, recall, f1


def un_nesesery(node, list_):
    if node.children is not None:
        for child in node.children.values():
            if child.feature_index in list_:
                list_.remove(child.feature_index)
            un_nesesery(child, list_)


def build_tree(ratio):
    print_(f" build_tree({ratio})")
    q1 = Q1()
    decision_tree = DecisionTree()
    data = q1.prepared_data
    data_train, data_test = q1.get_ratio_of_rows_shuffled(data, ratio)
    tree = decision_tree.fit(data_train, target)

    list_ = copy.deepcopy(list(data.columns))
    list_.remove(target)
    un_nesesery(tree, list_)
    print_(f'unusable fields : {list_}')

    accuracy, precision, recall, f1 = decision_tree.tree_error(tree, data_test)
    print_(f'gridy accuracy : {1-q1.df[target].mean()}')
    print_(f"Accuracy: {accuracy}")
    print_(f"Precision: {precision:.2f}")
    print_(f"Recall: {recall:.2f}")
    print_(f"F1 Score: {f1:.2f}")
    return accuracy


def tree_error(k):
    print_(f"Running tree_error({k})")
    q1 = Q1()
    decision_tree = DecisionTree()
    data = q1.prepared_data

    n_samples = len(data[target])
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    accuracy = []
    precision = []
    recall = []
    f1 = []
    list__ = None

    for i in range(k):
        print(f"[{i+1}/{k}]")
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples

        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        data_train = data.iloc[train_indices]
        data_test = data.iloc[test_indices]

        tree = decision_tree.fit(data_train, target)
        accuracy_, precision_, recall_, f1_ = decision_tree.tree_error(tree, data_test)

        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)
        f1.append(f1_)

        list_ = copy.deepcopy(list(data.columns))
        list_.remove(target)
        un_nesesery(tree, list_)

        if list__ is None:
            list__ = list_
        else:
            for col in list__[:]:
                if col not in list_:
                    list__.remove(col)

    print_(f'Unusable fields: {list__}')  # Print the unusable fields

    print_(f"Accuracy: {np.mean(accuracy):.2f}")
    print_(f"Precision: {np.mean(precision):.2f}")
    print_(f"Recall: {np.mean(recall):.2f}")
    print_(f"F1 Score: {np.mean(f1):.2f}")

    return np.mean(accuracy)


# Ensure target is correctly defined elsewhere in your code


def is_late(row_input):
    print_(f" is_late({row_input}))")
    q1 = Q1()
    data = q1.prepared_data
    columns = list(q1.df.columns)
    row_ = pd.DataFrame({columns[index]: [row_input[index]] for index in range(len(row_input))})
    row = q1.prepare_data(row_).iloc[0]
    decision_tree = DecisionTree()
    tree = decision_tree.fit(data, target)


    return decision_tree._predict(row, tree)


if __name__ == '__main__':
    debug = True
    build_tree(0.8)
    tree_error(4)
    print(is_late([1,7,'0800-0859',2,1,25,143,'Southwest Airlines Co.',13056,107363,5873,1903352,13382999,0.00006178236301460919,0.00009889412309998219,8,'McCarran International',36.08,-115.152,'NONE',0.0,0.0,0.0,65.0,2.91]))
