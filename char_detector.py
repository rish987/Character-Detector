"""
Filename: char_detector.py

Author: Rishikesh Vaishnav

Created:
24/03/2017

Description:
This program provides an easy-to-use environment where a user can draw a symbol
that represents an ASCII character, enter what character they drew, and save
the data representing their drawing and the number it is associated with it.
Additionaly, the user can use this data to train a logistic regression and
neural network model that can detect the character that was drawn.

These files will be saved in a folder in this program's directory called:
char_data/[character drawn]_saved

'character drawn' refers to the ASCII character the user associated this
character to when creating the file.

The data will be saved in this folder's subdirectory called:
data/ 

Data files will be named:
[num]_[character drawn]_data.dat

'num' refers to the next number after the maximum numbered file already in that
folder. Think of it as the ID number of this drawn character.

The data file will contain the following:
- On the first line, the ASCII character this number represents
- On the following lines, the list of pixel values drawn on the screen, 
  represented as 0s (not drawn) and 1s (drawn), starting from the upper left
  corner and going from left to right and top to bottom.

TODO TODO additional information relevant to logistic regression and neural 
network
"""
from Tkinter import *
import os

from numpy import *
from scipy.optimize import fmin_bfgs

# window size (both width and height)
WINDOW_WIDTH = 300;
WINDOW_HEIGHT = 400;

# padding of all components inside of window
PADDING = 5;

# minimum and maximum drawing canvas sizes
DC_MIN_SIZE = 10;
DC_MAX_SIZE = 100;

# drawing canvas size (both width and height)
# NOTE: must evenly divide DC_SIZE_PX
DC_SIZE = 20;

# drawing canvas size in pixels (both width and height)
DC_SIZE_PX = 200;

# bit size in pixels (both width and height)
BIT_SIZE_PX = DC_SIZE_PX / DC_SIZE;

# currently entered character
char = '';

# is the mouse currently being pressed?
mouse_pressed = False;

# the drawing canvas
drawing_canvas = 0;

# the drawing
drawing = [ [ False for col in range( 0, DC_SIZE ) ]
              for row in range( 0, DC_SIZE ) ];

# was nothing entered in the text area for the entered character?
nothing_entered = True;

# to store the parameters for each char, as determined by logistic regression
chars_params_log = {};

# detected char label
detected_char_label = 0;

def main ():
    """
    Sets up the canvas for drawing and using this program.
    """
    global drawing_canvas;
    global drawing;
    global detected_char_label;

    # set up the window with appropriate dimensions 
    window = Tk();
    window.wm_title( "Character Detector" );
    window.resizable( width=False, height=False );
    window.geometry( '{}x{}'.format( WINDOW_WIDTH, WINDOW_HEIGHT ) );
    window.configure( background='white' );

    # --- create the drawing canvas with its grid ---

    # initialize the canvas with constant dimensions
    drawing_canvas = Canvas( window, width=DC_SIZE_PX + 1, 
        height=DC_SIZE_PX + 1, background='white', highlightbackground='white', 
        highlightcolor='white' );
    
    # load the drawing canvas
    drawing_canvas.pack( padx=PADDING, pady=PADDING );

    redraw_canvas();

    drawing_canvas.bind( "<Motion>", move_mouse )
    drawing_canvas.bind( "<ButtonPress-1>", press_mouse )
    drawing_canvas.bind( "<ButtonRelease-1>", release_mouse )

    # ---

    # --- add field to: enter text ---

    # frame to hold number entering stuff
    char_panel = Frame( window, bg='white');

    # create label
    char_drawn_lbl = Label( char_panel, text='Character Drawn:', bg='white' );
    char_drawn_lbl.pack( side=LEFT );

    # character entered, trace it to limit it to 1 character 
    char_entered = StringVar()
    char_entered.trace( "w", lambda name, index, mode, sv=char_entered: 
        char_callback( char_entered ) );

    # to store character entered; make sure that field only accepts 
    # one character
    char_field = Entry( char_panel, textvariable=char_entered, width=1 );
    char_field.pack( side=LEFT );

    # display the char panel
    char_panel.pack( padx=PADDING, pady=PADDING );
 
    # ---

    # --- add buttons to: save image, clear canvas ---

    # frame to hold control panel widgets
    control_panel = Frame( window );

    # add save button
    save_btn = Button( control_panel, text='Save', command=save_button );
    save_btn.pack( side=LEFT );
    
    # add save button
    clear_btn = Button( control_panel, text='Clear', command=clear_button );
    clear_btn.pack( side=LEFT );

    # display the control panel
    control_panel.pack( padx=PADDING, pady=PADDING );

    # ---

    # --- add buttons for training ---

    train_panel = Frame( window );

    train_log_btn = Button( train_panel, text="Train this Char - Logistic", 
        command=train_log_button );
    train_log_btn.pack( side=LEFT );

    train_all_btn = Button( train_panel, text="Train All", 
        command=train_all_button );
    train_all_btn.pack( side=LEFT );

    # TODO add button for neural network

    train_panel.pack( padx=PADDING, pady=PADDING );

    # ---

    # --- add buttons for detection ---

    detect_panel = Frame( window );

    detect_log_btn = Button( detect_panel, text="Detect - Logistic", 
        command=detect_log_button );
    detect_log_btn.pack( side=LEFT );

    # TODO add button for neural network

    detect_panel.pack( padx=PADDING, pady=PADDING );

    # ---

    # --- add labels for detected character ---

    detected_char_panel = Frame( window );

    detected_char_info_label = Label( detected_char_panel, 
        text="Detected Char: ", bg='white' );
    detected_char_info_label.pack( side=LEFT );

    detected_char_label = Label( detected_char_panel, 
        text="N/A", bg='white' );
    detected_char_label.pack( side=LEFT );

    detected_char_panel.pack( padx=PADDING, pady=PADDING );

    # ---

    # --- set up the necessary folders ---

    # char folder does not already exist
    if not os.path.isdir( 'char_data' ):
        # make the char folder
        os.makedirs( 'char_data' );

    # ---

    # begin window loop
    window.mainloop();

def char_callback ( entered ):
    """
    Limits an entered character to 1 character.

    Parameters:
    entered - the character entered
    """
    global nothing_entered;

    # backspace was entered
    if len( entered.get() ) == 0:
        nothing_entered = True;

    # something was entered
    if len( entered.get() ) > 0:
        nothing_entered = False;

        # reset the current character to the first character in the field
        global char;
        char = entered.get()[ 0 ];

        # set the entered character to its own first character
        entered.set( char );

def press_mouse ( event ):
    """
    Handles pressing the mouse button by turning on a boolean that lets the
    program know that a mouse button is being pressed.

    Parameters: 
    event - the event that triggered this method
    """
    # turn on mouse pressed indicator boolean
    global mouse_pressed;
    mouse_pressed = True;

    # draw at this location
    draw_at( event.x, event.y );

def release_mouse ( event ):
    """
    Handles releasing the mouse button by turning off a boolean that lets the
    program know that a mouse button is being pressed.

    Parameters:
    event - the event that triggered this method
    """
    # turn off mouse pressed indicator boolean
    global mouse_pressed;
    mouse_pressed = False;

def move_mouse ( event ):
    """
    Handles moving the mouse button by drawing on the canvas if the mouse
    button is being pressed.

    Parameters:
    event - the event that triggered this method
    """
    # the mouse is being pressed
    if mouse_pressed:
        # draw at this location
        draw_at( event.x, event.y );

def clear_button ():
    """ 
    Handles the user pressing the clear button by clearing their drawing
    from the screen.
    """
    global drawing;

    # clear the drawing
    drawing = [ [ False for col in range( 0, DC_SIZE ) ]
                  for row in range( 0, DC_SIZE ) ];

    redraw_canvas();

def save_button ():
    """
    Handles the user pressing the save button by saving their drawing's image
    and data to file.
    """
    # check that the text field has one character in it to associate with
    # this drawing
    if nothing_entered:
        # don't save anything
        return;

    # --- check that the necessary folders exist ---

    # path to this char
    this_path = 'char_data/' + char + '_saved';

    # path to this char's data
    this_data_path = this_path + '/data';

    # check that the folders for this character do not already exist

    if not os.path.isdir( this_path ):
        os.makedirs( this_path );
    if not os.path.isdir( this_data_path ):
        os.makedirs( this_data_path );

    # ---

    # --- indicate that the training data file for this char is not 
    # up-to-date ---

    # path to char saved folder
    char_saved_path = 'char_data/' + char + '_saved';

    training_data_file_name = char_saved_path + '/' + char \
        + '_training_data_log.dat';

    # to write to the new training data file
    to_write = [ '0\n', '' ];

    # this training data already exists
    if os.path.exists( training_data_file_name ):
        training_data_file = open( training_data_file_name, 'r' );

        # read the lines in this file
        to_write = training_data_file.readlines();

        # change the first one to 0 to indicate that this data file is not
        # up-to-date
        to_write[ 0 ] = '0\n';


    training_data_file = open( training_data_file_name, 'w' );

    # write indicator to training file
    training_data_file.write( to_write[ 0 ] );
    training_data_file.write( to_write[ 1 ] );

    # ---

    # number to use for this file
    this_num = 1;

    # continue while the data file for this number exists
    while os.path.exists( this_data_path + '/' + str( this_num  )+ '_' + char 
    + '_' + 'data.dat' ):
        this_num = this_num + 1;

    # --- save the drawing's data ---

    # open file to write to
    data_file = open( this_data_path + '/' + str( this_num  )+ '_' + char 
    + '_' + 'data.dat', 'w' );

    # write this character to the data file
    data_file.write( char );
    data_file.write( '\n' );

    # go through all of the rows in the drawing
    for row in range( 0, len( drawing ) ):
        # go through all of the columns in this row of the drawing
        for col in range( 0, len( drawing[ row ] ) ):
            # this bit is on
            if drawing[ row ][ col ]:
                data_file.write( '1' )
            else:
                data_file.write( '0' )

    data_file.close();
    # ---

def train_log_button ():
    """
    Handles the user pressing the train logistic button by performing logistic
    regression to train the current character drawn.
    """
    global chars_params_log;

    # don't train if nothing was entered
    if nothing_entered:
        return;

    params = train_log_reg( char );

    # there was an issue training the current character
    if params == None:
        # don't add any params to the params list
        return;

    # train the current character drawn, and save this to the list
    chars_params_log[ char ] = params;

def train_all_button ():
    """
    Handles the user pressing the train all button by performing logistic
    regression to train all of the characters with saved data.
    """
    global chars_params_log;

    # get a list of all the subdirectories
    subdirs = [ x[ 0 ] for x in os.walk( 'char_data' ) ];

    # to store all of the characters with saved data
    chars = [];

    # go through all subdirectories
    for subdir in subdirs:
        # this is a folder ending in '_saved'
        if subdir.endswith( '_saved' ):
            # add this char to the list
            chars.append( subdir[ len( 'char_data/' ) ] );

    # go through all of the characters with saved data
    for this_char in chars:
        # train this character
        chars_params_log[ this_char ] = train_log_reg( this_char );
        # TODO train all neural networks

def detect_log_button ():
    """
    Handles the user pressing the detect logistic button by using parameters
    determined from previously performing logistic regression to detect the
    current character drawn.
    """
    global detected_char_label;

    # detect the current character drawn
    detected = detect_log_reg( chars_params_log );

    # reset the label to display the detected char
    detected_char_label.config( text=detected );

def draw_at ( x_loc, y_loc ):
    """
    Draws at the given x- and y-locations on the drawing canvas

    Parameters:
    x_loc - the x-location to draw at
    y_loc - the y-location to draw at
    """
    global drawing_canvas;
    global drawing;

    x_loc = x_loc - 2;
    y_loc = y_loc - 2;

    # the x- or y-location is out of range
    if x_loc >= DC_SIZE_PX or y_loc >= DC_SIZE_PX or x_loc < 0 or y_loc < 0:
        return;

    # x- and y-indices in the drawing to set
    x_ind = x_loc / BIT_SIZE_PX;
    y_ind = y_loc / BIT_SIZE_PX;

    # set this bit in the drawing
    drawing[ y_ind ][ x_ind ] = True;

    redraw_canvas();

def redraw_canvas ():
    """
    Redraws to canvas to fill in the boxes specified by the drawing.
    """
    global drawing_canvas;
    global drawing;

    # color the background
    drawing_canvas.create_rectangle( 1, 1, DC_SIZE_PX + 1, DC_SIZE_PX + 1,
        fill='white' );

    # number of lines to draw on the grid
    num_lines = DC_SIZE - 1;

    # offset of lines to draw on the grid
    lines_offset = BIT_SIZE_PX;

    # go through all gridlines to be drawn
    for line_i in range( 0, num_lines ):
        # the x and y pixel position of the line to be drawn
        pos = ( line_i + 1 ) * lines_offset;

        # draw the x and y gridlines
        drawing_canvas.create_line( 1, pos + 1, DC_SIZE_PX, pos + 1,
            fill='light grey' );
        drawing_canvas.create_line( pos + 1, 1, pos + 1, DC_SIZE_PX, 
            fill='light grey' );

    # draw the bounding rectangle of the grid
    drawing_canvas.create_rectangle( 1, 1, DC_SIZE_PX + 1, DC_SIZE_PX + 1,
        outline='black' );

    # go through all of the rows in the drawing
    for row in range( 0, len( drawing ) ):
        # go through all of the columns in this row of the drawing
        for col in range( 0, len( drawing[ row ] ) ):
            # this bit is on
            if drawing[ row ][ col ]:
                # draw this bit
                drawing_canvas.create_rectangle( 
                    BIT_SIZE_PX * col + 2, 
                    BIT_SIZE_PX * row + 2, 
                    BIT_SIZE_PX * col + BIT_SIZE_PX,
                    BIT_SIZE_PX * row + BIT_SIZE_PX,
                    fill='black');

def train_log_reg ( this_char ):
    """
    Performs regularized logistic regression to train and return a list of
    parameters that can be used in character detection for the given character.

    Parameters:
    this_char - char whose data will be used to train

    Returns:
    - a list containing the trained parameters for this character
    - -1 if this character does not have any data
    """
    # --- check if the parameters are already up-to-date ---

    # path to char saved folder
    char_saved_path = 'char_data/' + this_char + '_saved';

    prev_training_data_file = char_saved_path + '/' + this_char \
        + '_training_data_log.dat';

    # this saved data exists
    if os.path.exists( prev_training_data_file ):
        training_data_file = open( prev_training_data_file, 'r' );

        # read the lines in this file
        read = training_data_file.readlines();

        # the first line is a 1, so this data is already up-to-date
        if read[ 0 ] == '1\n':
            # convert the next line into a list of floats
            params = [ float( x ) for x in read[ 1 ].split() ];
            training_data_file.close();

            # return this list of floats
            return params;

        training_data_file.close();

    # ---

    # load the data
    chars_data = load_chars_data();

    # the data does not contain this char
    if not this_char in chars_data:
        # nothing to train
        return None;

    train_inps = [];
    train_outps = [];

    # --- format the data into matrix form ---

    # go through all of the keys in the data
    for key in chars_data:
        # the output for this key
        output = 0;
        # this key is the character we are training
        if key == this_char:
            # the output for this key should be 1
            output = 1;
            
        # go through all of the 
        for input in chars_data[ key ]:
            # set the input and output for this drawing
            train_inps.append( input );
            train_outps.append( output );

    train_inps_mx = matrix( train_inps );
    train_outps_mx = matrix( train_outps ).T;

    # ---

    # regularization parameter TODO find best value
    reg = 1;

    # number of features
    num_features = train_inps_mx.shape[ 1 ];

    # get the parameters
    params = fmin_bfgs( lambda x: get_reg_log_cost( x, train_inps_mx,
        train_outps_mx, reg ), zeros( num_features + 1 ), 
        fprime = lambda x: get_reg_log_grad( x, train_inps_mx, train_outps_mx, 
        reg ) );

    # --- save the data in a training data file ---

    # path to char saved folder
    char_saved_path = 'char_data/' + this_char + '_saved';

    training_data_file = open( char_saved_path + '/' + this_char + 
            '_training_data_log.dat', 
        'w' );

    # indicate that this training data file is up-to-date
    training_data_file.write( '1\n' );

    # go through all of the parameters
    for param in params:
        training_data_file.write( str( param ) + ' ' );

    training_data_file.close();

    # ---

    # train this character and return the trained parameters
    return params;

def get_reg_log_cost ( params_in, train_set_inps, train_set_outps, \
    reg_param ):
    """
    Returns the regularized logistic cost of a model, given the model's
    parameters and the training set inputs and outputs.

    Parameters:
    params_in (row vector of real numbers) - parameters of model
    train_set_inps_in ( matrix of real numbers ) - training set inputs, without
        intercept ( bias ) term
    train_set_outps_in ( vector of real numbers ) - training set outputs
    reg_param ( positive real number ) - regularization parameter
    """
    # convert params to column vector
    params = matrix( params_in ).T;

    # number of training examples
    m = float( len( train_set_inps ) );

    # training set inputs with bias term
    train_set_inps_adj = insert( train_set_inps, 0, 1, axis = 1 );

    # get the predicted outputs
    pred_outps = sigmoid( train_set_inps_adj * params );

    # list of individual costs for each training example
    costs = multiply( -train_set_outps, log( pred_outps ) ) - \
        multiply( ( 1 - train_set_outps ), log( 1 + 1e-15 - pred_outps ) );

    # cost, not including regularization
    cost = ( 1 / m ) * sum( costs )
    # cost, including regularization
    cost += ( reg_param / ( 2 * m ) ) * sum( multiply( params[ 1: ], 
        params[ 1: ] ) );

    # return the cost
    return cost;

def sigmoid ( data ):
    """
    Returns a matrix containg the sigmoid of each element in a given matrix of
    data.

    Parameters:
    data ( matrix of real numbers ) - the data to take the sigmoid of
    """
    # return the sigmoid of the data
    return 1 / ( 1 + exp( -data ) )

def get_reg_log_grad ( params_in, train_set_inps, train_set_outps, \
    reg_param  ):
    """
    Returns the regularized logistic gradient for a given set of parameters
    and training data.

    Parameters:
    params_in (row vector of real numbers) - parameters of model
    train_set_inps( matrix of real numbers ) - training set inputs, without
        intercept ( bias ) term
    train_set_outps( vector of real numbers ) - training set outputs
    reg_param ( positive real number ) - regularization parameter
    """
    # convert params to column vector
    params = matrix( params_in ).T;

    # size of training data
    m = float( len( train_set_inps ) );

    # training set inputs with bias terms
    adj_train_set_inps = insert( train_set_inps, 0, 1, axis = 1 );
    
    #( adj_train_set_inps.T * diffs ) / m get the predicted outputs
    pred_outps = sigmoid( adj_train_set_inps * params );

    # the differences
    diffs = pred_outps - train_set_outps;

    # the gradients, without regularization
    grads = ( adj_train_set_inps.T * diffs ) / m;
    # the gradients, with regularization
    grads[ 1: ] = grads[ 1: ] + ( reg_param / m ) * params[ 1: ];

    # return the flattened gradients
    return array( grads ).flatten();

def load_chars_data ():
    """
    Loads the data for each character into a dictionary, where each key
    contains a matrix containing the data for the character, and returns 
    this dictionary.

    Returns:
    - a dictionary where each key is a character and contains an mxn matrix
    containing the data for this character, where m is the number of separate
    saved data files for this character, and n is the number of bits in each
    data file. Each row contains the list of values of each bit in the data
    file (True for 1 and False for 0).
    - -1 if this character does not have any data
    """
    # directory in which to find data files
    data_dir = 'char_data';

    # the data files do not exist
    if not os.path.isdir( data_dir ):
        return;

    # dictionary to return
    data_dict = {};

    # go through all of the subdirectories
    for subdir, dirs, files in os.walk( data_dir ):
        # go through all of the files in this subdirectory
        for file in files:
            # this file is a data file
            if file.endswith( 'data.dat' ) and subdir.endswith( '/data' ):
                # open the data file and get its content
                data_file = open( subdir + '/' + file, 'r' );
                data_file_content = data_file.readlines();

                # the character drawn in this file
                this_char = data_file_content[ 0 ][ 0 ];

                # to store this drawing
                this_drawing = [];

                # go through all of the characters in the second line
                for drawing_char in data_file_content[ 1 ]:
                    this_drawing.append( ord( drawing_char  )- ord( '0' ) );

                # this key does not already exist in the dictonary
                if not this_char in data_dict:
                    # initialize this list
                    data_dict[ this_char ] = [];

                # add this data to the correct list
                data_dict[ this_char ].append( this_drawing );
            
    return data_dict;

def detect_log_reg ( chars_params ):
    """
    Uses the given, pre-trained parameters to detect the current character
    drawn.

    Parameters:
    chars_params - list of pre-trained parameters for use in detection

    Returns:
    - the detected character
    """
    global drawing;

    # probability of this drawing to be each of the characters trained so far
    probs = {};

    # --- flatten the drawing ---

    flattened_drawing = [];
    # go through all of the rows in the drawing
    for row in range( 0, len( drawing ) ):
        # go through all of the columns in this row of the drawing
        for col in range( 0, len( drawing[ row ] ) ):
            if drawing[ row ][ col ]:
                flattened_drawing.append( 1 );
            else:
                flattened_drawing.append( 0 );

    flattened_drawing_mx = insert( matrix( flattened_drawing ), 0, 1, 
        axis = 1 );

    # ---

    # go through all of the trained characters
    for key in chars_params:
        # get the probability of this character
        probs[ key ] = ( sigmoid( flattened_drawing_mx * matrix( 
            chars_params[ key ] ).T ) ).item( ( 0, 0 ) );

    # TODO
    print probs;
    # TODO

    # maximum probability char so far
    max_prob_char = -1;

    # maximum probability 
    max_prob = -1;

    # go through all of the probabilites for each character
    for key in probs:
        # this key's probability
        this_prob = probs[ key ];

        # this character has a higher probability than those searched before
        if this_prob > max_prob:
            max_prob = this_prob;
            max_prob_char = key;

    return max_prob_char;

# TODO TODO neural network stuff

# start the program
main();
