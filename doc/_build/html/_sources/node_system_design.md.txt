## Design of the node system

### Processor class 
- A function to create the GUI for displaying its internal state (the text inside should be updatable by the render callback)
- The `send` and `receve` function should be delegated to the base class so that we can swap implementation easily


### Manager
- Should have a manager class to query the connection between different module and make sure their data type matches