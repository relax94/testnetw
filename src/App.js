import React, {Component} from 'react';
import logo from './logo.svg';
import './App.css';
import {FlexContainer} from './FlexContainer'
import {SettingsComponent} from './SettingsComponent'
import {NN} from './core/network'

var nn = new NN({
    layers: {
        input : {
            size: 2
        },
        hidden : {
            size: 2
        },
        output : {
            size: 1
        },
    }
})

class App extends Component {
    render() {
        return (
            <div className="App">
                <div className="App-header">
                    <img src={logo} className="App-logo" alt="logo"/>
                    <h2>Welcome to  React Application</h2>
                </div>
                <FlexContainer>
                    <SettingsComponent/>
                </FlexContainer>
            </div>
    );
    }
    }

    export default App;
