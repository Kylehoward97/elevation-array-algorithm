'''CSC108: Assignment 2
Author: Kyle Howard
Student #: 1002350190
'''

from typing import List


THREE_BY_THREE = [[1, 2, 1],
                  [4, 6, 5],
                  [7, 8, 9]]

FOUR_BY_FOUR = [[1, 2, 6, 5],
                [4, 5, 3, 2],
                [7, 9, 8, 1],
                [1, 2, 1, 4]]

UNIQUE_3X3 = [[1, 2, 3],
              [9, 8, 7],
              [4, 5, 6]]

UNIQUE_4X4 = [[10, 2, 3, 30],
              [9, 8, 7, 11],
              [4, 5, 6, 12],
              [13, 14, 15, 16]]


def compare_elevations_within_row(elevation_map: List[List[int]], map_row: int,
                                  level: int) -> List[int]:
    '''Return a new list containing three counts: the number of elevations 
    from row number map_row of elevation_map that are less than, equal to, 
    and greater than elevation level.

    Precondition: elevation_map is a valid elevation map.
                  0 <= map_row < len(elevation_map).

    >>> compare_elevations_within_row(THREE_BY_THREE, 1, 5)
    [1, 1, 1]
    >>> compare_elevations_within_row(FOUR_BY_FOUR, 1, 2)
    [0, 1, 3]
    '''
    less = 0
    even = 0
    more = 0
    row = elevation_map[map_row]
    result = []
    for elevation in row:
        if elevation > level:
            more += 1
        elif elevation < level:
            less += 1
        else:
            even += 1

    result.append(less)
    result.append(even)
    result.append(more)
    return result 


def update_elevation(elevation_map: List[List[int]], start: List[int],
                     stop: List[int], delta: int) -> None:
    '''Modify elevation_map so that the elevation of each cell 
    between cells start and stop, inclusive, changes by amount  delta.

    Precondition: elevation_map is a valid elevation map.
                  start and stop are valid cells in elevation_map.
                  start and stop are in the same row or column or both.
                  If start and stop are in the same row,
                      start's column <=  stop's column.
                  If start and stop are in the same column,
                      start's row <=  stop's row.
                  elevation_map[i, j] + delta >= 1
                      for each cell [i, j] that will change.

    >>> THREE_BY_THREE_COPY = [[1, 2, 1],
    ...                        [4, 6, 5],
    ...                        [7, 8, 9]]
    >>> update_elevation(THREE_BY_THREE_COPY, [1, 0], [1, 1], -2)
    >>> THREE_BY_THREE_COPY
    [[1, 2, 1], [2, 4, 5], [7, 8, 9]]
    >>> FOUR_BY_FOUR_COPY = [[1, 2, 6, 5],
    ...                      [4, 5, 3, 2],
    ...                      [7, 9, 8, 1],
    ...                      [1, 2, 1, 4]]
    >>> update_elevation(FOUR_BY_FOUR_COPY, [1, 2], [3, 2], 1)
    >>> FOUR_BY_FOUR_COPY
    [[1, 2, 6, 5], [4, 5, 4, 2], [7, 9, 9, 1], [1, 2, 2, 4]]

    '''
    #if start and stop are in the same row
    if start[0] == stop[0]:
        for i in range(min(start[1], stop[1]), max(start[1], stop[1]) + 1):
            elevation_map[start[0]][i] = elevation_map[start[0]][i] + delta
        
    #if start and stop are in the same column  
    if start[1] == stop[1]:
        for i in range(min(start[0], stop[0]), max(start[0], stop[0]) + 1):
            elevation_map[i][start[1]] = elevation_map[i][start[1]] + delta
        
    #if start and stop are the same cell
    if start[0] == stop[0] and start[1] == stop[1]:
        elevation_map[start[0]][start[1]] = (elevation_map[start[0]][start[1]]
                                             + delta)
        
                                                                    
        


def get_average_elevation(elevation_map: List[List[int]]) -> float:
    '''Return the average elevation across all cells in elevation_map.

    Precondition: elevation_map is a valid elevation map.

    >>> get_average_elevation(UNIQUE_3X3)
    5.0
    >>> get_average_elevation(FOUR_BY_FOUR)
    3.8125
    '''
    numerator = 0
    denominator = 0
    for row in elevation_map:
        for index in row:
            denominator += 1
            numerator += index
    return numerator / denominator


def find_peak(elevation_map: List[List[int]]) -> List[int]:
    '''Return the cell that is the highest point in the elevation_map.

    Precondition: elevation_map is a valid elevation map.
                  Every elevation value in elevation_map is unique.

    >>> find_peak(UNIQUE_3X3)
    [1, 0]
    >>> find_peak(UNIQUE_4X4)
    [0, 3]
    '''
    highest_cell = [0, 0]
    current_max = 0
    i = -1
    for row in elevation_map:
        i += 1
        j = -1
        for index in row:
            j += 1
            if index > current_max:
                current_max = index
                highest_cell[0] = i
                highest_cell[1] = j
    return highest_cell
        
    


def is_sink(elevation_map: List[List[int]], cell: List[int]) -> bool:
    '''Return True if and only if cell exists in the elevation_map
    and cell is a sink.

    Precondition: elevation_map is a valid elevation map.
                  cell is a 2-element list.

    >>> is_sink(THREE_BY_THREE, [0, 5])
    False
    >>> is_sink(THREE_BY_THREE, [0, 2])
    True
    >>> is_sink(THREE_BY_THREE, [1, 1])
    False
    >>> is_sink(FOUR_BY_FOUR, [2, 3])
    True
    >>> is_sink(FOUR_BY_FOUR, [3, 2])
    True
    >>> is_sink(FOUR_BY_FOUR, [1, 3])
    False
    '''
    # finds if the cell is within the elevation_map
    
    matrix = [0, 0]
    n = -1
    for row in elevation_map:
        n += 1
        row = row
    matrix = [n, n]

    if cell[0] > matrix[0] or cell[1] > matrix[1]:
        return False
    if cell[0] < 0 or cell[1] < 0:
        return False

    #find value of cell
    cell_value = elevation_map[cell[0]][cell[1]]

    #check surrounding cells
    checker = 0
    for i in range(cell[0] - 1, cell[0] + 2):
        j = 0
        for j in range(cell[1] - 1, cell[1] + 2):

            if valid_cell(elevation_map, [i, j]) is True:

                if elevation_map[i][j] >= cell_value:
                    
                    checker = True
                else:
                    return False
            else:
                continue
    return checker







            
# helper function to find if cell is within matrix
def valid_cell(matrix: List[List[int]], x: List[int]) -> bool:  
    """Check to see if x is a proper cell within the given matrix
    >>> valid_cell(THREE_BY_THREE, [0, 5])
    False
    >>> valid_cell(THREE_BY_THREE, [0, 2])
    True
    >>> valid_cell(THREE_BY_THREE, [1, 1])
    True

    """
    dim = [0, 0]
    n = -1
    for row in matrix:
        n += 1
        row = row
    dim = [n, n]

    if x[0] > dim[0] or x[1] > dim[1]:
        return False
    if x[0] < 0 or x[1] < 0:
        return False
    return True
    

    
        

    

    

    
        

    
    


def find_local_sink(elevation_map: List[List[int]],
                    cell: List[int]) -> List[int]:
    '''Return the local sink of cell cell in elevation_map.

    Precondition: elevation_map is a valid elevation map.
                  elevation_map contains no duplicate elevation values.
                  cell is a valid cell in elevation_map.

    >>> find_local_sink(UNIQUE_3X3, [1, 1])
    [0, 0]
    >>> find_local_sink(UNIQUE_3X3, [2, 0])
    [2, 0]
    >>> find_local_sink(UNIQUE_4X4, [1, 3])
    [0, 2]
    >>> find_local_sink(UNIQUE_4X4, [2, 2])
    [2, 1]
    '''
    
    #check surrounding cells
    sink_value = elevation_map[cell[0]][cell[1]]
     
    sink_location = cell
    
    for i in range(cell[0] - 1, cell[0] + 2):
        j = 0
        
        for j in range(cell[1] - 1, cell[1] + 2):

            if valid_cell(elevation_map, [i, j]) is True:
            
                if elevation_map[i][j] <= sink_value:
                    sink_value = elevation_map[i][j]
                    sink_location[0] = i
                    sink_location[1] = j
                      
                else:
                    continue
            else:
                continue

    return sink_location


def can_hike_to(elevation_map: List[List[int]], start: List[int],
                dest: List[int], supplies: int) -> bool:
    '''Return True if and only if a hiker can move from start to dest in
    elevation_map without running out of supplies.

    Precondition: elevation_map is a valid elevation map.
                  start and dest are valid cells in elevation_map.
                  dest is North-West of start.
                  supplies >= 0

    >>> map = [[1, 6, 5, 6],
    ...        [2, 5, 6, 8],
    ...        [7, 2, 8, 1],
    ...        [4, 4, 7, 3]]
    >>> can_hike_to(map, [3, 3], [2, 2], 10)
    True
    >>> can_hike_to(map, [3, 3], [2, 2], 8)
    False
    >>> can_hike_to(map, [3, 3], [3, 0], 7)
    
    True
    >>> can_hike_to(map, [3, 3], [3, 0], 6)
    False
    >>> can_hike_to(map, [3, 3], [0, 0], 18)
    True
    >>> can_hike_to(map, [3, 3], [0, 0], 17)
    False
    '''

    current_cell_location = start
    current_cell_value = (elevation_map[current_cell_location[0]]
                          [current_cell_location[1]])

    north_cell_location = ([current_cell_location[0] - 1,
                            current_cell_location[1]])
    north_cell_value = (elevation_map[current_cell_location[0] - 1]
                        [current_cell_location[1]])

    west_cell_location = ([current_cell_location[0],
                           current_cell_location[1] -1])
    west_cell_value = (elevation_map[current_cell_location[0]]
                       [current_cell_location[1] - 1])

    while current_cell_location != dest:
    
        #move north if
        if (abs(north_cell_value - current_cell_value) <=
                abs(west_cell_value - current_cell_value)
                and west_cell_location[1] >= dest[1]
                and north_cell_location[0] >= dest[0]):
            
            #subtract supplies
            supplies -= abs(north_cell_value - current_cell_value)

            if supplies < 0:
                return False
            
            else:
                #move to new cell
                current_cell_location = north_cell_location
                current_cell_value = (elevation_map[current_cell_location[0]]
                                      [current_cell_location[1]])

                north_cell_location = ([current_cell_location[0] - 1,
                                        current_cell_location[1]])
                north_cell_value = (elevation_map[current_cell_location[0] - 1]
                                    [current_cell_location[1]])

                west_cell_location = ([current_cell_location[0],
                                       current_cell_location[1] -1])
                west_cell_value = (elevation_map[current_cell_location[0]]
                                   [current_cell_location[1] - 1])

                if current_cell_location == dest:
                    return True
                
        #move west if
        else:
            #subtract supplies
            supplies -= abs(west_cell_value - current_cell_value)

            if supplies < 0:
                return False
            
            else:
                #move to new cell
                current_cell_location = west_cell_location
                current_cell_value = (elevation_map[current_cell_location[0]]
                                      [current_cell_location[1]])

                north_cell_location = ([current_cell_location[0] - 1,
                                        current_cell_location[1]])
                north_cell_value = (elevation_map[current_cell_location[0] - 1]
                                    [current_cell_location[1]])

                west_cell_location = ([current_cell_location[0],
                                       current_cell_location[1] -1])
                west_cell_value = (elevation_map[current_cell_location[0]]
                                   [current_cell_location[1] - 1])

                if current_cell_location == dest:
                    return True
                
    
    

def get_lower_resolution(elevation_map: List[List[int]]) -> List[List[int]]:
    '''Return a new elevation map, which is constructed from the values
    of elevation_map by decreasing the number of points within it.

    Precondition: elevation_map is a valid elevation map.

    >>> get_lower_resolution(
    ...     [[1, 6, 5, 6],
    ...      [2, 5, 6, 8],
    ...      [7, 2, 8, 1],
    ...      [4, 4, 7, 3]])
    [[3, 6], [4, 4]]
    >>> get_lower_resolution(
    ...     [[7, 9, 1],
    ...      [4, 2, 1],
    ...      [3, 2, 3]])
    [[5, 1], [2, 3]]
    '''
    final_matrix = []

    n = dimentions(elevation_map)
    
    for i in range(0, n + 1, 2):#skips every other row
        new_row = []
        for j in range(0, n + 1, 2):#skips every other col
            new_row.append(segment_average(make_segment(elevation_map, i, j)))
            
        final_matrix.append(new_row)
    return final_matrix
            
            



# find dimention of a map
def dimentions(elevation_map: List[List[int]])-> int:
    """Finds the dimentions of the matrix.

    >>>dimentions(THREE_BY_THREE)
    2
    >>>dimentions(FOUR_BY_FOUR)
    3
    >>>dimentions(UNIQUE_3X3)
    2

    """
    n = -1
    for row in elevation_map:
        n += 1
        row = row
    return n
    

# make a segment
def make_segment(elevation_map: List[List[int]], row: int,
                 col: int) -> List[List[int]]:
    """Takes an elevation map at a point on the map
    and breaks it down into 2 x 2 matricies.
        

    >>>make_segment(THREE_BY_THREE, 0, 0)
    [[1, 2], [4, 6]]
    >>>make_segment(FOUR_BY_FOUR, 0, 0)
    [[1, 2], [4, 5]]
    >>>make_segment(THREE_BY_THREE, 0, 2)
    [[1], [5]]          

    """
    new_matrix = []

    #check if cell below exists
    if valid_cell(elevation_map, [row + 1, col]) is True:
                  
        for i in range(row, row + 2): # if it does
            segment = []
            new_matrix.append(segment)
        
            #check if cell to the right exists
            if valid_cell(elevation_map, [row, col+1]) is True: #if it does
            
                for j in range(col, col + 2):
                    segment.append(elevation_map[i][j])
                
            else: #if cell to right DNE
                for j in range(col, col + 1):
                    segment.append(elevation_map[i][j])

    else:#if cell below DNE

        for i in range(row, row + 1):
            segment = []
            new_matrix.append(segment)

            #check if cell to the right exists
            if valid_cell(elevation_map, [row, col+1]) is True: #if it does
            
                for j in range(col, col + 2):
                    segment.append(elevation_map[i][j])
                
            else: #if cell to right DNE
                for j in range(col, col + 1):
                    segment.append(elevation_map[i][j])
        

                
            
    return new_matrix



# take segment average
def segment_average(segment: List[List[int]]) -> List[int]:
    """Takes a given matrix and finds the aveage of
    all entries. Always rounds down.

    >>>segment_average([[1, 2], [3, 4]])
    2
    >>>segment_average([[3], [5]])
    4
    >>>segment_average([[5, 7], [4, 7]])
    5
    """
    average = 0
    div = 0
    num = 0
    
    for row in segment:

        for col in row:
            div += 1
            num += col

    average = int(num/div)
    
    return average
        
