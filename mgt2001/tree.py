from ete3 import Tree, faces, AttrFace, TreeStyle


def ts_layout(node, fgcolor='#FF4ba6', fsize=12):
    '''
    Please set up as the following code shows:

    ```py
    ts = TreeStyle()
    ts.mode = 'r' # 'c'
    ts.min_leaf_separation = 40
    # ts.branch_vertical_margin = 20
    # Do not add leaf names automatically
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.optimal_scale_level = 'mid'
    ts.margin_top = 5
    ts.margin_left = 5
    ts.margin_right = 5
    # Use my custom layout
    ts.layout_fn = ts_layout
    ```
    '''
    node.img_style["size"] = 2
    node.img_style["shape"] = "circle"
    node.img_style["fgcolor"] = fgcolor
    if node.is_leaf():
        # If terminal node, draws its name
        name_face = AttrFace("name", fsize=fsize)  # fgcolor="royalblue"
        name_face.margin_left = 3
        faces.add_face_to_node(name_face, node, column=0,
                               position="branch-right")
    elif not node.is_root():
        # If internal node, draws label with smaller font size
        name_face = AttrFace("name", fsize=fsize)
        name_face.margin_left = 2
        name_face.margin_bottom = 2
        # Adds the name face to the image at the preferred position
        faces.add_face_to_node(name_face, node, column=0,
                               position="branch-top")
        # Possible values are “branch-right”, “branch-top”, “branch-bottom”, “float”, “float-behind” and “aligned”.
