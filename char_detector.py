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

# window size (both width and height)
WINDOW_WIDTH = 250;
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

def main ():
    """
    Sets up the canvas for drawing and using this program.
    """
    global drawing_canvas;
    global drawing;

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

    train_log_btn = Button( train_panel, text="Train - Logistic", 
        command=train_log_button );
    train_log_btn.pack( side=LEFT );

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

    # number to use for this file
    this_num = 1;

    # continue while both the data and image file for this number exist
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

    # train the current character drawn, and save this to the list
    chars_params_log[ char ] = train_log_reg( char );

def detect_log_button ():
    """
    Handles the user pressing the detect logistic button by using parameters
    determined from previously performing logistic regression to detect the
    current character drawn.
    """
    # detect the current character drawn
    detect_log_reg( chars_params_log );

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
    # load the data
    chars_data = load_chars_data();

    # TODO TODO composite the data into two matrices, one with inputs and one
    # with output, for this charater

    # TODO TODO train this character

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
        return

    # go through all of the subdirectories
    for subdir, dirs, files in os.walk( data_dir ):
        # go through all of the files in this subdirectory
        for file in files:
            # this file is a data file
            if file.endswith( '.dat' ):
                # open the data file and get its content
                data_file = open( subdir + '/' + file, 'r' );
                data_file_content = data_file.readlines();

                # TODO
            
    # TODO TODO load the characters

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
    # TODO TODO detect the character

# TODO TODO neural network stuff

# start the program
main();
